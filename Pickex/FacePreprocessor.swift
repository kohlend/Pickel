//
//  FacePreprocessor.swift
//  Pickex
//
//  Photo -> aligned 112×112 CVPixelBuffer for the recognition model.
//
//  Detection + 5 keypoints come from Apple's Vision framework
//  (VNDetectFaceLandmarksRequest): deterministic, Apple-maintained, tuned per
//  device. The custom SCRFD Core ML detector was retired — its keypoint heads
//  were numerically unstable on-device (same photo, different keypoints per
//  run), which poisoned alignment no matter how the decode was fixed.
//
//  From Vision's landmark regions we take pupils, nose and mouth corners and
//  run the standard ArcFace 5-point similarity alignment. Every keypoint set
//  must pass an anatomy gate (left-of-right eyes/mouth, eyes above nose above
//  mouth, sane eye distance) before it may drive alignment — rotated or
//  degenerate sets are rejected, and a rotation-retry loop recovers faces in
//  sideways/lying-down photos.
//
//  The recognition model bakes its own normalization (y = x/127.5 − 1, RGB);
//  we hand it a plain 112×112 BGRA buffer and do NO pixel math here.
//

import CoreImage
import CoreVideo
import CoreGraphics
import ImageIO
import Vision

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
    private let ciContext: CIContext
    private let outputColorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    /// ArcFace 112×112 reference template (top-left origin):
    /// [image-left eye, image-right eye, nose, image-left mouth, image-right mouth].
    private static let arcFaceTemplateTopLeft: [CGPoint] = [
        CGPoint(x: 38.2946, y: 51.6963),
        CGPoint(x: 73.5318, y: 51.5014),
        CGPoint(x: 56.0252, y: 71.7366),
        CGPoint(x: 41.5493, y: 92.3655),
        CGPoint(x: 70.7299, y: 92.2041),
    ]

    public static let debugVersion = "v25-vision"

    public init(config: FacePreprocessorConfig = FacePreprocessorConfig()) {
        self.config = config
        self.ciContext = CIContext(options: [.cacheIntermediates: false])
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
        for face in faces {   // already sorted largest-first
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

    /// Debug: largest face's box + eye distance from the Vision path.
    public func debugEyeInfo(from cgImage: CGImage,
                             orientation: CGImagePropertyOrientation = .up) -> String {
        guard let (source, faces) = try? detectUsable(from: cgImage, orientation: orientation),
              let f = faces.first else { return "no usable face [vision-v1]" }
        let eyeDist = hypot(f.keypoints[1].x - f.keypoints[0].x, f.keypoints[1].y - f.keypoints[0].y)
        return String(format: "img %dx%d score=%.2f box[%.0f,%.0f %.0fx%.0f] eyeDist=%.0f [vision-v1]",
                      source.width, source.height, f.score,
                      f.bbox.origin.x, f.bbox.origin.y, f.bbox.width, f.bbox.height, eyeDist)
    }

    /// Debug: a ≤640px thumbnail of the photo with the largest usable face's
    /// box (green) and 5 keypoints (red) drawn on it.
    public func debugAnnotatedInput(from cgImage: CGImage,
                                    orientation: CGImagePropertyOrientation = .up) -> CGImage? {
        guard let (source, faces) = try? detectUsable(from: cgImage, orientation: orientation),
              let f = faces.first else { return nil }
        let W = CGFloat(source.width), H = CGFloat(source.height)
        let scale = min(1, 640 / max(W, H))
        let w = Int(W * scale), h = Int(H * scale)
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: outputColorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(source, in: CGRect(x: 0, y: 0, width: CGFloat(w), height: CGFloat(h)))
        // Top-left coords → this context's bottom-left coords.
        func toCtx(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * scale, y: (H - p.y) * scale) }
        ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1)); ctx.setLineWidth(2)
        let o = toCtx(CGPoint(x: f.bbox.minX, y: f.bbox.maxY))
        ctx.stroke(CGRect(x: o.x, y: o.y, width: f.bbox.width * scale, height: f.bbox.height * scale))
        ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        for k in f.keypoints {
            let c = toCtx(k)
            ctx.fillEllipse(in: CGRect(x: c.x - 4, y: c.y - 4, width: 8, height: 8))
        }
        return ctx.makeImage()
    }

    // MARK: detection (Vision)

    /// Detect faces + landmarks with Vision; keep only sets that pass the
    /// anatomy gate. If nothing usable is found upright, retry the image
    /// rotated right/left/180 so sideways (lying-down) faces still work.
    /// Returns the (possibly rotated) image the coordinates live in, with
    /// faces sorted largest-first.
    private func detectUsable(from cgImage: CGImage,
                              orientation: CGImagePropertyOrientation) throws -> (CGImage, [DetectedFace]) {
        let (base, _, _) = try upright(cgImage, orientation)
        var sawAnyFace = false
        for rotation in [CGImagePropertyOrientation.up, .right, .left, .down] {
            guard let (img, _, _) = try? upright(base, rotation) else { continue }
            let faces = visionFaces(in: img)
            sawAnyFace = sawAnyFace || !faces.isEmpty
            let usable = faces.filter(passesKeypointGate)
            if !usable.isEmpty { return (img, usable) }
        }
        throw sawAnyFace ? FacePreprocessError.degenerateLandmarks
                         : FacePreprocessError.noFaceFound
    }

    /// Vision faces → DetectedFace with the 5 ArcFace keypoints in top-left
    /// pixel coordinates, sorted largest-first.
    private func visionFaces(in source: CGImage) -> [DetectedFace] {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: source, options: [:])
        guard (try? handler.perform([request])) != nil else { return [] }
        let W = CGFloat(source.width), H = CGFloat(source.height)
        let size = CGSize(width: W, height: H)
        var faces: [DetectedFace] = []
        for obs in request.results ?? [] {
            guard let lm = obs.landmarks,
                  let lPupil = lm.leftPupil?.pointsInImage(imageSize: size).first,
                  let rPupil = lm.rightPupil?.pointsInImage(imageSize: size).first,
                  let lips = lm.outerLips?.pointsInImage(imageSize: size),
                  lips.count >= 3 else { continue }
            // Nose point: centroid of the nose region (robust against point
            // ordering; small offsets are absorbed by the 5-point LSQ fit).
            let nosePts = lm.nose?.pointsInImage(imageSize: size) ?? []
            guard !nosePts.isEmpty else { continue }
            let nose = CGPoint(x: nosePts.map(\.x).reduce(0, +) / CGFloat(nosePts.count),
                               y: nosePts.map(\.y).reduce(0, +) / CGFloat(nosePts.count))
            let mouthL = lips.min { $0.x < $1.x }!
            let mouthR = lips.max { $0.x < $1.x }!
            // Order eyes by image x ourselves — never trust left/right naming.
            let eyeL = lPupil.x <= rPupil.x ? lPupil : rPupil
            let eyeR = lPupil.x <= rPupil.x ? rPupil : lPupil
            // Vision returns BOTTOM-LEFT-origin pixel coords; flip to top-left.
            func tl(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x, y: H - p.y) }
            let r = VNImageRectForNormalizedRect(obs.boundingBox, Int(W), Int(H))
            let bbox = CGRect(x: r.minX, y: H - r.maxY, width: r.width, height: r.height)
            faces.append(DetectedFace(bbox: bbox,
                                      keypoints: [tl(eyeL), tl(eyeR), tl(nose), tl(mouthL), tl(mouthR)],
                                      score: obs.confidence))
        }
        return faces.sorted { $0.bbox.width * $0.bbox.height > $1.bbox.width * $1.bbox.height }
    }

    /// Anatomy gate (top-left coords). Rejects rotated, collapsed or otherwise
    /// scrambled keypoint sets — the source of every garbage crop so far:
    /// minimum face size, sane eye distance relative to the box, left-of-right
    /// eyes and mouth corners, eyes above nose, nose above mouth.
    private func passesKeypointGate(_ face: DetectedFace) -> Bool {
        guard min(face.bbox.width, face.bbox.height) >= 48 else { return false }
        let k = face.keypoints
        let eyeDist = hypot(k[1].x - k[0].x, k[1].y - k[0].y)
        guard eyeDist >= 5, eyeDist >= 0.20 * face.bbox.width else { return false }
        guard k[0].x < k[1].x, k[3].x < k[4].x else { return false }
        guard max(k[0].y, k[1].y) < k[2].y, k[2].y < min(k[3].y, k[4].y) else { return false }
        return true
    }

    // MARK: internals

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
