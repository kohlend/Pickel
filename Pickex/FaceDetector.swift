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
import ImageIO

public struct DetectedFace: Sendable {
    public let bbox: CGRect          // original-image pixels, top-left origin
    public let keypoints: [CGPoint]  // 5 points, original-image pixels, top-left origin
    public let score: Float
}

public final class SCRFDDetector: @unchecked Sendable {

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
        let out: MLFeatureProvider
        do {
            let input = try MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
            out = try model.prediction(from: input)
        } catch { return "predict THREW: \(error)" }
        var lines = ["img \(W)x\(H) scale=\(String(format: "%.3f", scale))"]
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
