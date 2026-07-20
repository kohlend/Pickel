//
//  FacePreprocessor.swift
//  Pickex
//
//  Photo -> aligned 112×112 CVPixelBuffer for the recognition model.
//
//  Detection + 5 keypoints come from the SCRFD FaceDetector (Core ML), NOT
//  Vision — Vision's landmarks proved unreliable/nondeterministic and could not
//  drive alignment. SCRFD's keypoints match InsightFace exactly (validated),
//  so we do a proper 5-point similarity alignment to the ArcFace template, the
//  same as training. That is what makes recognition robust to lighting/angle.
//
//  The model bakes its own normalization (y = x/127.5 − 1, RGB); we hand it a
//  plain 112×112 BGRA buffer and do NO pixel math here.
//

import CoreImage
import CoreVideo
import CoreGraphics
import ImageIO

public enum FacePreprocessError: Error {
    case noFaceFound
    case detectorUnavailable
    case degenerateLandmarks
    case pixelBufferCreationFailed
    case renderFailed
}

public struct FacePreprocessorConfig {
    /// Side length the recognition model expects.
    public let outputSize: Int = 112
    /// Max faces returned by detection per photo.
    public var maxFaces: Int = 8
    public init() {}
}

public final class FacePreprocessor {

    private let config: FacePreprocessorConfig
    private let detector: SCRFDDetector?
    private let ciContext: CIContext
    private let outputColorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    /// ArcFace 112×112 reference template (top-left origin):
    /// [left-eye, right-eye, nose, left-mouth, right-mouth].
    private static let arcFaceTemplateTopLeft: [CGPoint] = [
        CGPoint(x: 38.2946, y: 51.6963),
        CGPoint(x: 73.5318, y: 51.5014),
        CGPoint(x: 56.0252, y: 71.7366),
        CGPoint(x: 41.5493, y: 92.3655),
        CGPoint(x: 70.7299, y: 92.2041),
    ]

    public static let debugVersion = "v23-cpudet"

    /// Loads the SCRFD detector from the app bundle ("FaceDetector.mlpackage")
    /// unless one is injected. Non-throwing so it can be a default argument;
    /// detection surfaces `.detectorUnavailable` if the model is missing.
    public init(config: FacePreprocessorConfig = FacePreprocessorConfig(),
                detector: SCRFDDetector? = nil) {
        self.config = config
        self.ciContext = CIContext(options: [.cacheIntermediates: false])
        if let detector {
            self.detector = detector
        } else if let url = Bundle.main.url(forResource: "FaceDetectorModel", withExtension: "mlmodelc")
                    ?? Bundle.main.url(forResource: "FaceDetector", withExtension: "mlmodelc") {
            self.detector = try? SCRFDDetector(modelURL: url)
        } else {
            self.detector = nil
        }
    }

    public struct FaceCropResult {
        public let pixelBuffer: CVPixelBuffer
        public let faceCount: Int
        public let alignedWithLandmarks: Bool
    }

    // MARK: Public API

    public func makeFaceInput(from cgImage: CGImage,
                              orientation: CGImagePropertyOrientation = .up) throws -> CVPixelBuffer {
        try makeFaceInputDetailed(from: cgImage, orientation: orientation).pixelBuffer
    }

    public func makeFaceInputDetailed(from cgImage: CGImage,
                                      orientation: CGImagePropertyOrientation = .up) throws -> FaceCropResult {
        let (source, faces) = try detectUsable(from: cgImage, orientation: orientation)
        let H = CGFloat(source.height)
        // Largest usable face first.
        let byArea = faces.sorted { $0.bbox.width*$0.bbox.height > $1.bbox.width*$1.bbox.height }
        for face in byArea {
            if let buffer = try? align(face, source: source, height: H) {
                return FaceCropResult(pixelBuffer: buffer, faceCount: faces.count, alignedWithLandmarks: true)
            }
        }
        throw FacePreprocessError.degenerateLandmarks
    }

    /// Crop EVERY usable detected face (largest first). Faces that fail
    /// alignment are dropped individually. Throws `.noFaceFound` when the
    /// photo has no detectable face in any rotation.
    public func makeAllFaceInputs(from cgImage: CGImage,
                                  orientation: CGImagePropertyOrientation = .up,
                                  maxFaces: Int = 8) throws -> [FaceCropResult] {
        let (source, faces) = try detectUsable(from: cgImage, orientation: orientation)
        let H = CGFloat(source.height)
        return faces.prefix(max(1, maxFaces)).compactMap { face in
            guard let b = try? align(face, source: source, height: H) else { return nil }
            return FaceCropResult(pixelBuffer: b, faceCount: faces.count, alignedWithLandmarks: true)
        }
    }

    public func makeFaceInput(from ciImage: CIImage,
                              orientation: CGImagePropertyOrientation = .up) throws -> CVPixelBuffer {
        guard let cg = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            throw FacePreprocessError.renderFailed
        }
        return try makeFaceInput(from: cg, orientation: orientation)
    }

    public func makeFaceInputOrNil(from cgImage: CGImage,
                                   orientation: CGImagePropertyOrientation = .up) -> CVPixelBuffer? {
        try? makeFaceInput(from: cgImage, orientation: orientation)
    }

    /// Debug: largest face's score + 5 keypoints + box (SCRFD).
    public func debugEyeInfo(from cgImage: CGImage,
                             orientation: CGImagePropertyOrientation = .up) -> String {
        guard let (source, _, _) = try? upright(cgImage, orientation),
              let d = detector else { return "detector unavailable" }
        let faces = d.detect(source, maxFaces: 1)
        guard let raw = faces.first else { return "no face [\(SCRFDDetector.buildTag)]" }
        let f = refineKeypoints(raw, in: source, using: d)
        let eyeDist = hypot(f.keypoints[1].x - f.keypoints[0].x, f.keypoints[1].y - f.keypoints[0].y)
        return String(format: "img %dx%d score=%.2f box[%.0f,%.0f %.0fx%.0f] eyeDist=%.0f [%@]",
                      source.width, source.height, f.score,
                      f.bbox.origin.x, f.bbox.origin.y, f.bbox.width, f.bbox.height, eyeDist,
                      SCRFDDetector.buildTag)
    }

    /// Debug: the annotated 640×640 detector input for the given photo, with
    /// the REFINED keypoints drawn (the ones alignment actually uses).
    public func debugAnnotatedInput(from cgImage: CGImage,
                                    orientation: CGImagePropertyOrientation = .up) -> CGImage? {
        guard let (source, _, _) = try? upright(cgImage, orientation),
              let d = detector else { return nil }
        let refined = d.detect(source, maxFaces: 1).first
            .map { refineKeypoints($0, in: source, using: d) }
        return d.debugAnnotatedInput(source, face: refined)
    }

    /// Debug: raw detector diagnostics (predict ok/throw, output shapes, max
    /// scores) for the largest-image path. Surfaces WHY detection is empty.
    public func debugDetectorDiagnostics(from cgImage: CGImage,
                                         orientation: CGImagePropertyOrientation = .up) -> String {
        guard let (source, _, _) = try? upright(cgImage, orientation) else { return "upright failed" }
        guard let d = detector else { return "detector unavailable (model not in bundle?)" }
        return d.diagnostics(source)
    }

    // MARK: internals

    /// SCRFD only handles near-upright faces: lying-down/rotated faces either
    /// go undetected or come back with collapsed keypoints. So: bake EXIF
    /// orientation, detect; if nothing usable, retry the image rotated 90°
    /// left/right, then 180°. Returns the (possibly rotated) image the face
    /// coordinates live in, plus only the faces whose keypoints pass the gate.
    private func detectUsable(from cgImage: CGImage,
                              orientation: CGImagePropertyOrientation) throws -> (CGImage, [DetectedFace]) {
        guard let detector else { throw FacePreprocessError.detectorUnavailable }
        let (base, _, _) = try upright(cgImage, orientation)
        var sawAnyFace = false
        for rotation in [CGImagePropertyOrientation.up, .right, .left, .down] {
            guard let (img, _, _) = try? upright(base, rotation) else { continue }
            let faces = detector.detect(img, maxFaces: config.maxFaces)
            sawAnyFace = sawAnyFace || !faces.isEmpty
            let refined = faces.map { refineKeypoints($0, in: img, using: detector) }
            let usable = refined.filter(passesKeypointGate)
            if !usable.isEmpty { return (img, usable) }
        }
        throw sawAnyFace ? FacePreprocessError.degenerateLandmarks
                         : FacePreprocessError.noFaceFound
    }

    /// Two-stage keypoint refinement. The detector's keypoints are only
    /// reliable when the face is ~230–400px in the 640 frame (measured: 231,
    /// 293 and 394 give eyeDist ratio ~0.42–0.47; tiny faces AND huge selfie
    /// faces both collapse onto one spot). The box, however, is right at every
    /// size. So: stage 1 finds the box; stage 2 renders the face at exactly
    /// 320px into a fresh 640 canvas — the proven sweet spot — re-detects, and
    /// maps box + keypoints back into full-image coordinates.
    private func refineKeypoints(_ face: DetectedFace, in source: CGImage,
                                 using detector: SCRFDDetector) -> DetectedFace {
        let side: CGFloat = 640, target: CGFloat = 320
        let bw = max(face.bbox.width, face.bbox.height)
        guard bw > 1 else { return face }
        let s = target / bw
        let W = CGFloat(source.width), H = CGFloat(source.height)
        // Place the box center at the canvas center (CG bottom-left space).
        let x0 = side / 2 - s * face.bbox.midX
        let y0 = side / 2 - s * (H - face.bbox.midY)
        guard let ctx = CGContext(data: nil, width: Int(side), height: Int(side),
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: outputColorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue) else { return face }
        ctx.interpolationQuality = .high
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
        ctx.draw(source, in: CGRect(x: x0, y: y0, width: W * s, height: H * s))
        guard let canvas = ctx.makeImage(),
              let best = detector.detect(canvas, maxFaces: 1).first,
              // Must be OUR face: near the canvas center, not some neighbor.
              abs(best.bbox.midX - side / 2) < target / 2,
              abs(best.bbox.midY - side / 2) < target / 2 else { return face }
        // Canvas top-left coords → full-image top-left coords.
        let ox = -x0 / s, oy = H - (side - y0) / s
        func back(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x / s + ox, y: p.y / s + oy) }
        let origin = back(best.bbox.origin)
        let bb = CGRect(x: origin.x, y: origin.y,
                        width: best.bbox.width / s, height: best.bbox.height / s)
        return DetectedFace(bbox: bb, keypoints: best.keypoints.map(back),
                            score: max(face.score, best.score))
    }

    /// Frontal faces have eyeDist ≈ 0.35–0.45 of the box width; collapsed
    /// keypoints (sideways/strong-profile/tiny faces) fall way below and would
    /// align into a garbage nose-zoom crop that matches everything. Also
    /// requires a minimum face size in source pixels: a 112px crop upscaled
    /// from a sub-48px face is unrecognizable mush, and mush embeddings
    /// cluster with each other — better to skip such faces entirely.
    private func passesKeypointGate(_ face: DetectedFace) -> Bool {
        guard min(face.bbox.width, face.bbox.height) >= 48 else { return false }
        let eyeDist = hypot(face.keypoints[1].x - face.keypoints[0].x,
                            face.keypoints[1].y - face.keypoints[0].y)
        return eyeDist >= 5 && eyeDist >= 0.20 * face.bbox.width
    }

    /// Upright CGImage (bakes EXIF orientation) + its size. Detection and
    /// rendering both use this single space.
    private func upright(_ cgImage: CGImage,
                         _ orientation: CGImagePropertyOrientation) throws -> (CGImage, CGFloat, CGFloat) {
        if orientation == .up {
            return (cgImage, CGFloat(cgImage.width), CGFloat(cgImage.height))
        }
        let oriented = CIImage(cgImage: cgImage).oriented(orientation)
        let normalized = oriented.transformed(
            by: CGAffineTransform(translationX: -oriented.extent.origin.x, y: -oriented.extent.origin.y))
        guard let cg = ciContext.createCGImage(normalized, from: normalized.extent) else {
            throw FacePreprocessError.renderFailed
        }
        return (cg, CGFloat(cg.width), CGFloat(cg.height))
    }

    /// 5-point similarity alignment of one face onto the ArcFace template.
    private func align(_ face: DetectedFace, source: CGImage, height H: CGFloat) throws -> CVPixelBuffer {
        guard passesKeypointGate(face) else {
            throw FacePreprocessError.degenerateLandmarks
        }
        // Keypoints are top-left pixel coords; convert to the render space
        // (bottom-left, y-up) used by CGContext. Template likewise.
        let src = face.keypoints.map { CGPoint(x: $0.x, y: H - $0.y) }
        let dst = Self.arcFaceTemplateTopLeft.map {
            CGPoint(x: $0.x, y: CGFloat(config.outputSize) - $0.y)
        }
        let transform = try similarityTransform(from: src, to: dst)
        return try render(source, transform: transform)
    }

    /// Least-squares 2-D similarity transform (rotation + uniform scale +
    /// translation) mapping src → dst. Solves (a,b,tx,ty) in
    /// X = a·x − b·y + tx, Y = b·x + a·y + ty.
    private func similarityTransform(from src: [CGPoint], to dst: [CGPoint]) throws -> CGAffineTransform {
        var A = [[Double]](repeating: [Double](repeating: 0, count: 4), count: 4)
        var c = [Double](repeating: 0, count: 4)
        func addRow(_ r: [Double], _ t: Double) {
            for i in 0..<4 { c[i] += r[i]*t; for j in 0..<4 { A[i][j] += r[i]*r[j] } }
        }
        for k in 0..<src.count {
            let x = Double(src[k].x), y = Double(src[k].y)
            addRow([x, -y, 1, 0], Double(dst[k].x))
            addRow([y,  x, 0, 1], Double(dst[k].y))
        }
        guard let p = solve4x4(A, c) else { throw FacePreprocessError.degenerateLandmarks }
        return CGAffineTransform(a: CGFloat(p[0]), b: CGFloat(p[1]),
                                 c: CGFloat(-p[1]), d: CGFloat(p[0]),
                                 tx: CGFloat(p[2]), ty: CGFloat(p[3]))
    }

    /// Warp `source` by `transform` into a 112×112 BGRA buffer via CGContext
    /// (unambiguous bottom-left, y-up semantics matching the transform math).
    private func render(_ source: CGImage, transform: CGAffineTransform) throws -> CVPixelBuffer {
        let size = config.outputSize
        guard let buffer = makePixelBuffer(width: size, height: size) else {
            throw FacePreprocessError.pixelBufferCreationFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                  width: size, height: size, bitsPerComponent: 8,
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
        ctx.draw(source, in: CGRect(x: 0, y: 0, width: source.width, height: source.height))
        return buffer
    }

    private func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:],
        ]
        var pb: CVPixelBuffer?
        let ok = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                     kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        return ok == kCVReturnSuccess ? pb : nil
    }

    private func solve4x4(_ Ain: [[Double]], _ bin: [Double]) -> [Double]? {
        var A = Ain, b = bin
        for col in 0..<4 {
            var piv = col
            for r in (col+1)..<4 where abs(A[r][col]) > abs(A[piv][col]) { piv = r }
            if abs(A[piv][col]) < 1e-12 { return nil }
            if piv != col { A.swapAt(piv, col); b.swapAt(piv, col) }
            for r in 0..<4 where r != col {
                let f = A[r][col]/A[col][col]
                if f == 0 { continue }
                for k in col..<4 { A[r][k] -= f*A[col][k] }
                b[r] -= f*b[col]
            }
        }
        return (0..<4).map { b[$0]/A[$0][$0] }
    }
}
