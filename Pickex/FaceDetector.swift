//
//  FaceDetector.swift
//  Pickex
//
//  SCRFD face detector (InsightFace det_500m) as Core ML. Returns face boxes
//  AND 5 reliable keypoints (eyes/nose/mouth) per face — the keypoints Vision
//  couldn't provide reliably. These drive proper 5-point alignment, which is
//  what makes recognition robust to lighting/angle (validated: same decode as
//  InsightFace, keypoint diff 0.0).
//
//  Input: 640×640 letterboxed image (RGB, (x−127.5)/128 baked into the model).
//

import CoreML
import CoreVideo
import CoreGraphics
import CoreImage
import ImageIO

public struct DetectedFace: Sendable {
    public let bbox: CGRect          // original-image pixels, top-left origin
    public let keypoints: [CGPoint]  // 5 points, original-image pixels, top-left origin
    public let score: Float
}

public final class SCRFDDetector: @unchecked Sendable {

    /// Bump when FaceDetector.swift changes, so the debug UI proves the new
    /// file is actually compiled in (the version marker lives in a different
    /// file and can't confirm this one).
    public static let buildTag = "det-v2-interp"

    private let model: MLModel
    private let inputSize = 640
    private let scoreThreshold: Float = 0.5
    private let nmsThreshold: CGFloat = 0.4
    private let strides = [8, 16, 32]
    private let numAnchors = 2
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    public init(model: MLModel) { self.model = model }

    public convenience init(modelURL: URL,
                            configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        self.init(model: try MLModel(contentsOf: modelURL, configuration: configuration))
    }

    /// Detect faces, largest first. Coordinates are in the ORIGINAL image space.
    public func detect(_ cgImage: CGImage, maxFaces: Int = 8) -> [DetectedFace] {
        let W = cgImage.width, H = cgImage.height
        let scale = CGFloat(inputSize) / CGFloat(max(W, H))   // fit long side to 640
        guard let buffer = letterbox(cgImage, scale: scale) else { return [] }

        let out: MLFeatureProvider
        do {
            let input = try MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
            out = try model.prediction(from: input)
        } catch { return [] }

        // The converted model's output feature names are NOT the assumed
        // s8/b8/k8 (it was torch-traced, so coremltools named them generically).
        // Classify each output by SHAPE instead of name: the row count
        // N = grid²·anchors identifies the stride, and the channel count
        // C ∈ {1,4,10} identifies score / bbox / keypoints. Fully name-agnostic.
        var nToStride: [Int: Int] = [:]
        for stride in strides {
            let grid = inputSize / stride
            nToStride[grid * grid * numAnchors] = stride
        }
        var scoreByStride: [Int: [Float]] = [:]
        var bboxByStride: [Int: [Float]] = [:]
        var kpsByStride: [Int: [Float]] = [:]
        for name in model.modelDescription.outputDescriptionsByName.keys {
            guard let arr = out.featureValue(for: name)?.multiArrayValue else { continue }
            let dims = arr.shape.map { $0.intValue }
            guard let n = dims.first(where: { nToStride[$0] != nil }),
                  let stride = nToStride[n] else { continue }
            let vals = floats(arr)
            switch arr.count / n {
            case 1: scoreByStride[stride] = vals
            case 4: bboxByStride[stride] = vals
            case 10: kpsByStride[stride] = vals
            default: continue
            }
        }

        var faces: [DetectedFace] = []
        for stride in strides {
            guard let sc = scoreByStride[stride],
                  let bp = bboxByStride[stride],
                  let kp = kpsByStride[stride] else { continue }
            let grid = inputSize / stride
            let fStride = Float(stride)
            var row = 0
            for y in 0..<grid {
                for x in 0..<grid {
                    for _ in 0..<numAnchors {
                        defer { row += 1 }
                        if sc[row] < scoreThreshold { continue }
                        let cx = Float(x) * fStride, cy = Float(y) * fStride
                        let x1 = cx - bp[row*4+0]*fStride, y1 = cy - bp[row*4+1]*fStride
                        let x2 = cx + bp[row*4+2]*fStride, y2 = cy + bp[row*4+3]*fStride
                        var pts: [CGPoint] = []
                        for p in 0..<5 {
                            let px = (cx + kp[row*10+p*2]*fStride) / Float(scale)
                            let py = (cy + kp[row*10+p*2+1]*fStride) / Float(scale)
                            pts.append(CGPoint(x: CGFloat(px), y: CGFloat(py)))
                        }
                        let rect = CGRect(x: CGFloat(x1)/scale, y: CGFloat(y1)/scale,
                                          width: CGFloat(x2-x1)/scale, height: CGFloat(y2-y1)/scale)
                        faces.append(DetectedFace(bbox: rect, keypoints: pts, score: sc[row]))
                    }
                }
            }
        }
        return Array(nms(faces).prefix(maxFaces))
    }

    /// Diagnostic: run the model once and report predict success/throw, the
    /// runtime output shapes, and the max score per output. Lets us see WHY
    /// detection returns nothing without a Mac to run Core ML on.
    public func diagnostics(_ cgImage: CGImage) -> String {
        let W = cgImage.width, H = cgImage.height
        let scale = CGFloat(inputSize) / CGFloat(max(W, H))
        guard let buffer = letterbox(cgImage, scale: scale) else { return "letterbox failed" }
        // Is the letterbox buffer actually non-black? Mean of BGRA bytes over
        // the region the image was drawn into. 0 => letterbox produced black.
        let bufMean = meanLuma(buffer)
        let out: MLFeatureProvider
        do {
            let input = try MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
            out = try model.prediction(from: input)
        } catch { return "predict THREW: \(error)" }
        // Sanity: does the output depend on the input at all? Feed a solid-white
        // 640×640 buffer; if its scores match the real buffer's, the converted
        // model is broken (outputs independent of input).
        var whiteMax = Float(-1)
        if let white = solidBuffer(gray: 255),
           let wout = try? model.prediction(from:
               MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: white)])) {
            whiteMax = ["s8", "s16", "s32"].compactMap {
                wout.featureValue(for: $0)?.multiArrayValue
            }.flatMap { floats($0) }.max() ?? -1
        }
        var lines = ["img \(W)x\(H) scale=\(String(format: "%.3f", scale)) bufMean=\(String(format: "%.1f", bufMean)) whiteScoreMax=\(String(format: "%.3f", whiteMax))"]
        for name in out.featureNames.sorted() {
            guard let a = out.featureValue(for: name)?.multiArrayValue else {
                lines.append("\(name): not a multiArray"); continue
            }
            let f = floats(a)
            let mx = f.max() ?? -1
            lines.append("\(name) \(a.shape.map { $0.intValue }) n=\(a.count) dt=\(a.dataType.rawValue) max=\(String(format: "%.3f", mx))")
        }
        return lines.joined(separator: "\n")
    }

    /// A 640×640 BGRA buffer filled with a single gray level (for input probes).
    private func solidBuffer(gray: UInt8) -> CVPixelBuffer? {
        let side = inputSize
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:],
        ]
        var pb: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, side, side,
                                  kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let g = CGFloat(gray) / 255
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb),
                                  width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: g, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
        return pb
    }

    /// Mean byte value across the pixel buffer (BGRA). 0 => black.
    private func meanLuma(_ pb: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return -1 }
        let h = CVPixelBufferGetHeight(pb)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        let p = base.bindMemory(to: UInt8.self, capacity: bpr * h)
        var sum = 0.0, count = 0
        // Sample every 16th pixel to stay cheap.
        for y in stride(from: 0, to: h, by: 4) {
            for x in stride(from: 0, to: bpr, by: 64) {
                sum += Double(p[y * bpr + x]); count += 1
            }
        }
        return count > 0 ? sum / Double(count) : -1
    }

    /// Debug: the exact 640×640 image fed to the model, with the top face's
    /// box (green) and 5 keypoints (red) drawn on it. If the image is a sharp
    /// face but the red dots pile up on one spot → model/keypoint bug. If the
    /// image is aliased mush → the letterbox downscale is the culprit.
    public func debugAnnotatedInput(_ cgImage: CGImage) -> CGImage? {
        let W = cgImage.width, H = cgImage.height
        let scale = CGFloat(inputSize) / CGFloat(max(W, H))
        guard let buffer = letterbox(cgImage, scale: scale) else { return nil }
        let faces = detect(cgImage, maxFaces: 1)   // decode in original coords
        let side = inputSize
        guard let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        // Draw the letterbox buffer as the background.
        let ci = CIImage(cvPixelBuffer: buffer)
        if let bg = CIContext().createCGImage(ci, from: ci.extent) {
            ctx.draw(bg, in: CGRect(x: 0, y: 0, width: side, height: side))
        }
        // Original coords → 640 letterbox coords: multiply by scale. CGContext
        // is bottom-left origin, keypoints/box are top-left → flip y.
        func toCtx(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * scale, y: CGFloat(side) - p.y * scale)
        }
        if let f = faces.first {
            ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1)); ctx.setLineWidth(2)
            let o = toCtx(CGPoint(x: f.bbox.minX, y: f.bbox.maxY))
            ctx.stroke(CGRect(x: o.x, y: o.y, width: f.bbox.width * scale, height: f.bbox.height * scale))
            ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
            for k in f.keypoints {
                let c = toCtx(k)
                ctx.fillEllipse(in: CGRect(x: c.x - 4, y: c.y - 4, width: 8, height: 8))
            }
        }
        return ctx.makeImage()
    }

    // MARK: helpers

    private func floats(_ a: MLMultiArray) -> [Float] {
        let n = a.count
        if a.dataType == .float32 {
            let p = a.dataPointer.bindMemory(to: Float32.self, capacity: n)
            return Array(UnsafeBufferPointer(start: p, count: n))
        }
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n { out[i] = a[i].floatValue }
        return out
    }

    /// Draw the image scaled by `scale` into the top-left of a 640×640 black
    /// buffer (aspect preserved), matching the Python letterbox.
    private func letterbox(_ cgImage: CGImage, scale: CGFloat) -> CVPixelBuffer? {
        let side = inputSize
        let nw = CGFloat(cgImage.width) * scale, nh = CGFloat(cgImage.height) * scale
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:],
        ]
        var pb: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, side, side,
                                  kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb),
                                  width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
        // High-quality resampling is essential: phone photos are ~5000px and
        // get scaled ~16× into the 640 input. With CG's default interpolation
        // that heavy downscale aliases away fine detail — the box still
        // detects but the eye/nose/mouth keypoints collapse (eyes land on the
        // same pixel), which wrecks alignment. .high uses a proper kernel.
        ctx.interpolationQuality = .high
        // CG is bottom-left origin: place the image in the TOP-left region.
        ctx.draw(cgImage, in: CGRect(x: 0, y: CGFloat(side) - nh, width: nw, height: nh))
        return pb
    }

    private func nms(_ faces: [DetectedFace]) -> [DetectedFace] {
        let sorted = faces.sorted { $0.score > $1.score }
        var keep: [DetectedFace] = []
        for f in sorted {
            if keep.allSatisfy({ iou($0.bbox, f.bbox) < nmsThreshold }) { keep.append(f) }
        }
        return keep.sorted { $0.bbox.width * $0.bbox.height > $1.bbox.width * $1.bbox.height }
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull { return 0 }
        let ia = inter.width * inter.height
        return ia / (a.width*a.height + b.width*b.height - ia)
    }
}
