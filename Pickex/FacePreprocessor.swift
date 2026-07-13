//
//  FacePreprocessor.swift
//  Pickex
//
//  Turns a photo (CGImage / CIImage, e.g. from a PHAsset) into the exact
//  112×112 CVPixelBuffer that the MobileFaceNet Core ML model expects.
//
//  ── Model spec (see MobileFaceNet-CoreML/README.md §2) ──────────────────────
//  • Input  "face_image": IMAGE, RGB, 112×112.
//  • Output "embedding":  MultiArray Float32 [1, 512].
//  • Normalization  y = x/127.5 − 1  is BAKED INTO the .mlpackage
//    (ImageType scale = 1/127.5, bias = −1, RGB order).
//    → We must NOT normalize here. We hand Core ML a plain 112×112 image
//      buffer; Core ML does the RGB conversion and the scale/bias itself.
//      Normalizing again in Swift would silently corrupt every embedding
//      (model still runs, no error, just garbage output).
//  • The model was trained on InsightFace 5-point *similarity-aligned* faces,
//    so we align the detected face to the ArcFace 112×112 template. A plain
//    bounding-box crop is only the fallback (systematically misaligned →
//    degraded embeddings).
//
//  Pixel format: kCVPixelFormatType_32BGRA — the canonical buffer for a Core ML
//  ImageType input. Core ML maps BGRA → the model's RGB internally.
//
//  Assumptions made where the model doc left things open:
//  • ArcFace 5-point template coordinates (the InsightFace de-facto standard)
//    are used as the alignment target.
//  • Vision's per-eye landmark centroid (or pupil, when present) is used for
//    the eye points, the nose-region centroid for the nose, and the extreme
//    outer-lip points for the mouth corners. Eyes/mouth are ordered by image-x
//    so the mapping to the template is stable regardless of head roll.
//

import CoreImage
import CoreVideo
import CoreGraphics
import ImageIO
import Vision

// MARK: - Errors & configuration

public enum FacePreprocessError: Error {
    case noFaceFound            // Vision found no face — the common case for library photos.
    case landmarksUnavailable   // face found but no usable landmarks and fallback disabled.
    case degenerateLandmarks    // landmarks collinear/degenerate → transform not solvable.
    case invalidCrop            // crop rect empty after clamping.
    case pixelBufferCreationFailed
    case renderFailed
}

public struct FacePreprocessorConfig {
    /// Bounding-box crop margin *per side*, as a fraction of the box size.
    /// Default 0.0 — measured (margin×separation sweep): more margin shrinks
    /// the face in the 112 crop and collapses different identities together.
    /// 0.0 gave same/different GAP +0.33 vs +0.10 at 0.25. Squaring the box
    /// already adds context on the short side.
    public var faceMarginFraction: CGFloat = 0.0

    /// If true, fall back to a bounding-box+margin crop when landmarks are
    /// missing. If false, such faces throw `.landmarksUnavailable`.
    public var allowBoundingBoxFallback: Bool = true

    /// Master switch for landmark-based alignment. Default OFF: Vision's
    /// landmarks proved unreliable and nondeterministic in practice (same face
    /// yielding inter-eye distances from 5 px to 945 px), which produced
    /// broken aligned crops and near-random cross-photo matching. With this
    /// off we always use the bounding-box crop, which is consistent.
    public var useLandmarkAlignment: Bool = true

    /// (Only when useLandmarkAlignment is true) eye-landmark sanity gate: the
    /// detected inter-eye distance must be at least this fraction of the
    /// face-box width to trust the landmarks, else the bbox crop is used.
    public var minEyeDistanceToBoxRatio: CGFloat = 0.32

    /// Side length the model expects.
    public let outputSize: Int = 112

    public init() {}
}

// MARK: - FacePreprocessor

public final class FacePreprocessor {

    private let config: FacePreprocessorConfig
    /// Reusable GPU-backed context — created once, used for every photo. Doing
    /// the resample on Core Image (not UIKit) keeps the scan hot-path fast.
    private let ciContext: CIContext
    private let outputColorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    /// ArcFace 112×112 reference template, origin **top-left** (as published by
    /// InsightFace): [left-eye, right-eye, nose, left-mouth, right-mouth],
    /// where left/right are in *image* space (smaller-x first).
    private static let arcFaceTemplateTopLeft: [CGPoint] = [
        CGPoint(x: 38.2946, y: 51.6963),
        CGPoint(x: 73.5318, y: 51.5014),
        CGPoint(x: 56.0252, y: 71.7366),
        CGPoint(x: 41.5493, y: 92.3655),
        CGPoint(x: 70.7299, y: 92.2041),
    ]

    /// Bumped on every alignment-logic change so the demo UI can prove which
    /// code version is actually running (stale-build debugging).
    public static let debugVersion = "v8-align+margin0+flip"

    public init(config: FacePreprocessorConfig = FacePreprocessorConfig()) {
        self.config = config
        self.ciContext = CIContext(options: [.cacheIntermediates: false])
    }

    /// Debug helper: raw detection/eye geometry for the largest face, as text.
    public func debugEyeInfo(from cgImage: CGImage,
                             orientation: CGImagePropertyOrientation = .up) -> String {
        let W = CGFloat(cgImage.width), H = CGFloat(cgImage.height)
        guard let faces = try? detectFaces(in: cgImage, orientation: orientation),
              let largest = faces.max(by: {
                  $0.boundingBox.width * $0.boundingBox.height <
                  $1.boundingBox.width * $1.boundingBox.height
              }) else { return "detect: no face" }
        let bb = largest.boundingBox
        guard let eyes = eyeCenters(of: largest, imageWidth: W, imageHeight: H) else {
            return "eyes: unavailable"
        }
        let d = hypot(eyes.right.x - eyes.left.x, eyes.right.y - eyes.left.y)
        return String(format: "img %.0fx%.0f box[%.2f,%.2f %.2fx%.2f] eL(%.0f,%.0f) eR(%.0f,%.0f) dist=%.1f",
                      W, H, bb.origin.x, bb.origin.y, bb.width, bb.height,
                      eyes.left.x, eyes.left.y, eyes.right.x, eyes.right.y, d)
    }

    // MARK: Public API

    /// Result of cropping, with the detection metadata the reference-profile /
    /// scan steps need (how many faces were in the photo, whether we aligned
    /// via landmarks or fell back to the bounding box).
    public struct FaceCropResult {
        public let pixelBuffer: CVPixelBuffer
        public let faceCount: Int
        public let alignedWithLandmarks: Bool
    }

    /// Detect the largest face, align it to 112×112 and return a BGRA buffer
    /// ready to feed straight into the Core ML model (no further normalization).
    /// Throws a typed error (never crashes) so the scan pipeline can treat
    /// "no face on this photo" as the normal case.
    public func makeFaceInput(from cgImage: CGImage,
                              orientation: CGImagePropertyOrientation = .up) throws -> CVPixelBuffer {
        try makeFaceInputDetailed(from: cgImage, orientation: orientation).pixelBuffer
    }

    /// Same as `makeFaceInput` but also returns the detection metadata
    /// (`faceCount`, `alignedWithLandmarks`). Detection runs once.
    public func makeFaceInputDetailed(from cgImage: CGImage,
                                      orientation: CGImagePropertyOrientation = .up) throws -> FaceCropResult {
        let (image, W, H) = uprightImage(from: cgImage, orientation: orientation)
        let faces = try detectFaces(in: cgImage, orientation: orientation)
        let largest = faces.max {
            $0.boundingBox.width * $0.boundingBox.height <
            $1.boundingBox.width * $1.boundingBox.height
        }!
        let (buffer, aligned) = try alignAndRender(largest, in: image, W: W, H: H)
        return FaceCropResult(pixelBuffer: buffer, faceCount: faces.count, alignedWithLandmarks: aligned)
    }

    /// Crop EVERY detected face (largest first, capped at `maxFaces`) — used by
    /// the library scan, where the person we're looking for is often NOT the
    /// largest face in a group photo. Faces that can't be aligned or rendered
    /// are dropped individually rather than failing the whole photo.
    /// Throws only `.noFaceFound` (the normal no-face-photo case).
    public func makeAllFaceInputs(from cgImage: CGImage,
                                  orientation: CGImagePropertyOrientation = .up,
                                  maxFaces: Int = 8) throws -> [FaceCropResult] {
        let (image, W, H) = uprightImage(from: cgImage, orientation: orientation)
        let faces = try detectFaces(in: cgImage, orientation: orientation)
        let bySize = faces.sorted {
            $0.boundingBox.width * $0.boundingBox.height >
            $1.boundingBox.width * $1.boundingBox.height
        }
        var results: [FaceCropResult] = []
        for face in bySize.prefix(max(1, maxFaces)) {
            guard let (buffer, aligned) = try? alignAndRender(face, in: image, W: W, H: H) else { continue }
            results.append(FaceCropResult(pixelBuffer: buffer,
                                          faceCount: faces.count,
                                          alignedWithLandmarks: aligned))
        }
        return results
    }

    // Shared plumbing for the single-face and all-faces paths.

    private func uprightImage(from cgImage: CGImage,
                              orientation: CGImagePropertyOrientation) -> (CIImage, CGFloat, CGFloat) {
        // Work entirely in an upright image space so Vision's coordinates and
        // our render coordinates line up. `.oriented` gives an upright CIImage;
        // we normalize its extent origin to (0,0).
        let orientedRaw = CIImage(cgImage: cgImage).oriented(orientation)
        let image = orientedRaw.transformed(
            by: CGAffineTransform(translationX: -orientedRaw.extent.origin.x,
                                  y: -orientedRaw.extent.origin.y))
        return (image, image.extent.width, image.extent.height)
    }

    /// Landmark alignment with bbox fallback for one face; returns the buffer
    /// and whether landmarks were used.
    ///
    /// Alignment uses ONLY the two eye centers (Vision's most reliable
    /// landmarks): two point pairs determine the 4-DOF similarity transform
    /// exactly. An earlier version least-squares-fitted five points, but
    /// Vision's nose/mouth heuristics are unstable and the fit "compensated"
    /// with wild zoom/rotation. Validated against InsightFace's 5-point
    /// norm_crop: embeddings of eyes-only crops agree at cosine 0.97-0.98 and
    /// same/different-person separation is unchanged (see Pickex/README.md).
    private func alignAndRender(_ face: VNFaceObservation,
                                in image: CIImage, W: CGFloat, H: CGFloat) throws -> (CVPixelBuffer, Bool) {
        // Landmarks can be junk (simulator CPU path: nondeterministic points,
        // eye distances ~60% of plausible). Gate on the eye-distance/box-width
        // ratio (see config.minEyeDistanceToBoxRatio); junk falls back to the
        // known-good bbox crop, real landmarks get the aligned crop.
        let faceBoxWidthPx = face.boundingBox.width * W
        if config.useLandmarkAlignment,
           let eyes = eyeCenters(of: face, imageWidth: W, imageHeight: H),
           hypot(eyes.right.x - eyes.left.x, eyes.right.y - eyes.left.y)
               > max(8, config.minEyeDistanceToBoxRatio * faceBoxWidthPx) {
            let transform = eyePairTransform(from: eyes)
            return (try render(image, transform: transform), true)
        }
        guard config.allowBoundingBoxFallback else {
            throw FacePreprocessError.landmarksUnavailable
        }
        let transform = try boundingBoxTransform(box: face.boundingBox, imageWidth: W, imageHeight: H)
        return (try render(image, transform: transform), false)
    }

    public func makeFaceInput(from ciImage: CIImage,
                              orientation: CGImagePropertyOrientation = .up) throws -> CVPixelBuffer {
        guard let cg = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            throw FacePreprocessError.renderFailed
        }
        return try makeFaceInput(from: cg, orientation: orientation)
    }

    /// Non-throwing convenience — returns nil on any failure (incl. no face).
    public func makeFaceInputOrNil(from cgImage: CGImage,
                                   orientation: CGImagePropertyOrientation = .up) -> CVPixelBuffer? {
        try? makeFaceInput(from: cgImage, orientation: orientation)
    }

    // MARK: Step 1 — detection + landmarks

    /// Runs Vision face+landmark detection; throws `.noFaceFound` when empty.
    private func detectFaces(in cgImage: CGImage,
                             orientation: CGImagePropertyOrientation) throws -> [VNFaceObservation] {
        let request = VNDetectFaceLandmarksRequest()
        #if targetEnvironment(simulator)
        // The iOS SIMULATOR can fail Vision's NN-based requests with
        // VNErrorDomain Code=9 "Could not create inference context" when no
        // GPU inference context is available. Force the CPU path there.
        // (Deprecated API, but confined to simulator builds; real devices
        // never take this branch.)
        request.usesCPUOnly = true
        #endif
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
        try handler.perform([request])
        guard let faces = request.results, !faces.isEmpty else {
            throw FacePreprocessError.noFaceFound
        }
        return faces
    }

    /// Both eye centers in CI coordinates (origin bottom-left, y-up), ordered
    /// by image-x (left = smaller x) so the mapping is roll-invariant.
    /// Prefers the pupil when Vision provides it, else the eye-region centroid.
    ///
    /// Uses `pointsInImage(imageSize:)` — Vision's official conversion to
    /// image coordinates (lower-left origin, matching CI space). Manually
    /// mapping `normalizedPoints` via the bounding box mis-scaled the eye
    /// distance ~3-4× (massively zoomed crops); this API removes any
    /// ambiguity about what the normalized values are relative to.
    private func eyeCenters(of face: VNFaceObservation,
                            imageWidth W: CGFloat, imageHeight H: CGFloat) -> (left: CGPoint, right: CGPoint)? {
        guard let lm = face.landmarks,
              let leftEyeRegion = lm.leftEye,
              let rightEyeRegion = lm.rightEye else { return nil }

        let size = CGSize(width: W, height: H)
        func center(_ region: VNFaceLandmarkRegion2D?) -> CGPoint? {
            guard let pts = region?.pointsInImage(imageSize: size), !pts.isEmpty else { return nil }
            return centroid(pts)
        }

        guard let eyeA = center(lm.leftPupil) ?? center(leftEyeRegion),
              let eyeB = center(lm.rightPupil) ?? center(rightEyeRegion) else { return nil }
        return eyeA.x <= eyeB.x ? (eyeA, eyeB) : (eyeB, eyeA)
    }

    // MARK: Step 2 — geometry

    /// Exact similarity transform (rotation + uniform scale + translation)
    /// mapping the detected eye pair onto the ArcFace template's eye points
    /// (flipped to CI y-up coordinates). Two point pairs fully determine the
    /// 4 degrees of freedom — no least squares, no unstable extra landmarks.
    private func eyePairTransform(from eyes: (left: CGPoint, right: CGPoint)) -> CGAffineTransform {
        let side = CGFloat(config.outputSize)
        let tL = CGPoint(x: Self.arcFaceTemplateTopLeft[0].x, y: side - Self.arcFaceTemplateTopLeft[0].y)
        let tR = CGPoint(x: Self.arcFaceTemplateTopLeft[1].x, y: side - Self.arcFaceTemplateTopLeft[1].y)

        let sv = CGPoint(x: eyes.right.x - eyes.left.x, y: eyes.right.y - eyes.left.y)
        let dv = CGPoint(x: tR.x - tL.x, y: tR.y - tL.y)
        let scale = hypot(dv.x, dv.y) / hypot(sv.x, sv.y)
        let angle = atan2(dv.y, dv.x) - atan2(sv.y, sv.x)
        let a = scale * cos(angle)
        let b = scale * sin(angle)
        // Solve translation so that eyes.left maps exactly onto tL:
        // x' = a·x − b·y + tx ,  y' = b·x + a·y + ty
        let tx = tL.x - (a * eyes.left.x - b * eyes.left.y)
        let ty = tL.y - (b * eyes.left.x + a * eyes.left.y)
        return CGAffineTransform(a: a, b: b, c: -b, d: a, tx: tx, ty: ty)
    }

    /// (Retained for reference/tests) least-squares 2-D similarity transform
    /// mapping `src` → `dst` — unused by the main path since the eyes-only
    /// alignment replaced the 5-point fit.
    private func similarityTransform(from src: [CGPoint], to dst: [CGPoint]) throws -> CGAffineTransform {
        precondition(src.count == dst.count && src.count >= 2)
        // Normal equations A·p = c  (A is 4×4 symmetric, p = [a,b,tx,ty]).
        var A = [[Double]](repeating: [Double](repeating: 0, count: 4), count: 4)
        var c = [Double](repeating: 0, count: 4)
        func addRow(_ r: [Double], _ target: Double) {
            for i in 0..<4 {
                c[i] += r[i] * target
                for j in 0..<4 { A[i][j] += r[i] * r[j] }
            }
        }
        for k in 0..<src.count {
            let x = Double(src[k].x), y = Double(src[k].y)
            let X = Double(dst[k].x), Y = Double(dst[k].y)
            addRow([x, -y, 1, 0], X)   // X equation
            addRow([y,  x, 0, 1], Y)   // Y equation
        }
        guard let p = solve4x4(A, c) else { throw FacePreprocessError.degenerateLandmarks }
        let a = p[0], b = p[1], tx = p[2], ty = p[3]
        // CGAffineTransform: X = a·x + c·y + tx, Y = b·x + d·y + ty
        return CGAffineTransform(a: CGFloat(a), b: CGFloat(b),
                                 c: CGFloat(-b), d: CGFloat(a),
                                 tx: CGFloat(tx), ty: CGFloat(ty))
    }

    /// Fallback: expand the Vision bounding box by the margin, make it square,
    /// clamp to the image (edge faces don't crash), then map that square onto
    /// the 112×112 output (scale + translate, no rotation).
    private func boundingBoxTransform(box: CGRect, imageWidth W: CGFloat, imageHeight H: CGFloat) throws -> CGAffineTransform {
        // Vision box is normalized, origin bottom-left. Convert to CI pixels.
        var rect = CGRect(x: box.origin.x * W, y: box.origin.y * H,
                          width: box.width * W, height: box.height * H)

        let mx = rect.width * config.faceMarginFraction
        let my = rect.height * config.faceMarginFraction
        rect = rect.insetBy(dx: -mx, dy: -my)

        // Square, centered on the (expanded) box.
        var side = max(rect.width, rect.height)
        side = min(side, W, H)                       // can't exceed the image
        let cx = rect.midX, cy = rect.midY
        var originX = cx - side / 2
        var originY = cy - side / 2
        // Clamp so the square stays fully inside the image.
        originX = min(max(originX, 0), W - side)
        originY = min(max(originY, 0), H - side)

        guard side > 1 else { throw FacePreprocessError.invalidCrop }

        let s = CGFloat(config.outputSize) / side
        // Map (originX, originY) → (0,0) then scale to 112.
        return CGAffineTransform(scaleX: s, y: s).translatedBy(x: -originX, y: -originY)
    }

    // MARK: Step 3 + 5 — resample & pack into a CVPixelBuffer

    private func render(_ image: CIImage, transform: CGAffineTransform) throws -> CVPixelBuffer {
        // CGContext-based warp. Its semantics are unambiguous and match the
        // transform math exactly (bitmap origin bottom-left, y-up, source
        // pixel p lands at transform(p)). The previous CIContext.render path
        // produced mis-scaled output for rotated similarity transforms even
        // though axis-aligned (bbox) transforms rendered correctly.
        let size = config.outputSize
        guard let cg = ciContext.createCGImage(image, from: image.extent) else {
            throw FacePreprocessError.renderFailed
        }
        guard let buffer = makePixelBuffer(width: size, height: size) else {
            throw FacePreprocessError.pixelBufferCreationFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                  width: size, height: size,
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: outputColorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw FacePreprocessError.renderFailed
        }
        ctx.interpolationQuality = .high
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        ctx.concatenate(transform)
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
        return buffer
    }

    private func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:],
        ]
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA,
                                         attrs as CFDictionary, &pb)
        return status == kCVReturnSuccess ? pb : nil
    }

    // MARK: helpers

    private func centroid(_ pts: [CGPoint]) -> CGPoint {
        guard !pts.isEmpty else { return .zero }
        let n = CGFloat(pts.count)
        return CGPoint(x: pts.reduce(0) { $0 + $1.x } / n,
                       y: pts.reduce(0) { $0 + $1.y } / n)
    }

    /// Gaussian elimination with partial pivoting for a 4×4 system. Returns nil
    /// if (near-)singular.
    private func solve4x4(_ Ain: [[Double]], _ bin: [Double]) -> [Double]? {
        var A = Ain, b = bin
        let n = 4
        for col in 0..<n {
            var pivot = col
            for r in (col + 1)..<n where abs(A[r][col]) > abs(A[pivot][col]) { pivot = r }
            if abs(A[pivot][col]) < 1e-12 { return nil }
            if pivot != col { A.swapAt(pivot, col); b.swapAt(pivot, col) }
            let d = A[col][col]
            for r in 0..<n where r != col {
                let f = A[r][col] / d
                if f == 0 { continue }
                for k in col..<n { A[r][k] -= f * A[col][k] }
                b[r] -= f * b[col]
            }
        }
        var x = [Double](repeating: 0, count: n)
        for i in 0..<n { x[i] = b[i] / A[i][i] }
        return x
    }
}
