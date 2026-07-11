//
//  FaceEmbedder.swift
//  Pickex
//
//  The reusable single-image → 512-d embedding unit: preprocess (FacePreprocessor)
//  → run the MobileFaceNet Core ML model → read the "embedding" output.
//  Used both by the reference-profile builder (Step B) and the library scan
//  (Step C), so it lives on its own.
//
//  The model bakes its own normalization (see FacePreprocessor / model README),
//  so we feed the raw 112×112 buffer straight in. This type returns the RAW
//  512-d vector (not L2-normalized) plus how many faces were in the photo;
//  normalization happens where cosine similarity is used.
//

import CoreML
import CoreVideo
import CoreGraphics
import ImageIO

public enum FaceEmbeddingError: Error {
    case predictionFailed(Error)
    case outputMissing(String)        // named output not present
    case unexpectedOutputSize(Int)    // != expected dimensionality
}

public final class FaceEmbedder: @unchecked Sendable {
    // MLModel prediction and CIContext rendering are thread-safe, so a single
    // FaceEmbedder can be shared across concurrent tasks (TaskGroup in Step B,
    // parallel scan in Step C).

    public struct Result {
        public let vector: [Float]        // raw 512-d embedding
        public let faceCount: Int         // faces detected in the source photo
        public let alignedWithLandmarks: Bool
    }

    private let model: MLModel
    private let preprocessor: FacePreprocessor
    private let inputName: String
    private let outputName: String
    private let expectedDimension: Int

    /// Inject an already-loaded MLModel (e.g. the Xcode-generated
    /// `FaceEmbedding().model`) plus the shared preprocessor.
    public init(model: MLModel,
                preprocessor: FacePreprocessor = FacePreprocessor(),
                inputName: String = "face_image",
                outputName: String = "embedding",
                expectedDimension: Int = 512) {
        self.model = model
        self.preprocessor = preprocessor
        self.inputName = inputName
        self.outputName = outputName
        self.expectedDimension = expectedDimension
    }

    /// Load the model from a compiled/`.mlpackage` URL.
    public convenience init(modelURL: URL,
                            configuration: MLModelConfiguration = MLModelConfiguration(),
                            preprocessor: FacePreprocessor = FacePreprocessor(),
                            inputName: String = "face_image",
                            outputName: String = "embedding",
                            expectedDimension: Int = 512) throws {
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)
        self.init(model: model, preprocessor: preprocessor,
                  inputName: inputName, outputName: outputName,
                  expectedDimension: expectedDimension)
    }

    // MARK: single-image embedding (the reusable unit)

    /// Detect the largest face, preprocess, and run the model.
    /// Throws `FacePreprocessError` (incl. `.noFaceFound`) or `FaceEmbeddingError`.
    public func embedding(from cgImage: CGImage,
                          orientation: CGImagePropertyOrientation = .up) throws -> Result {
        let crop = try preprocessor.makeFaceInputDetailed(from: cgImage, orientation: orientation)
        let vector = try runModel(on: crop.pixelBuffer)
        return Result(vector: vector,
                      faceCount: crop.faceCount,
                      alignedWithLandmarks: crop.alignedWithLandmarks)
    }

    /// Embed a pre-cropped 112×112 buffer directly (e.g. a cached crop). Skips
    /// detection; used by the scan step when the crop already exists.
    public func embedding(fromCroppedBuffer buffer: CVPixelBuffer) throws -> [Float] {
        try runModel(on: buffer)
    }

    /// Embed EVERY face in the photo (largest first, capped at `maxFaces`).
    /// The library scan uses this: on group photos the person we're looking
    /// for is often not the largest face. Throws `.noFaceFound` when the photo
    /// has no faces (the normal case for most library photos).
    public func embeddingsForAllFaces(in cgImage: CGImage,
                                      orientation: CGImagePropertyOrientation = .up,
                                      maxFaces: Int = 8) throws -> [[Float]] {
        let crops = try preprocessor.makeAllFaceInputs(from: cgImage,
                                                       orientation: orientation,
                                                       maxFaces: maxFaces)
        // A face whose model run fails is dropped, not fatal for the photo.
        return crops.compactMap { try? runModel(on: $0.pixelBuffer) }
    }

    // MARK: model plumbing

    private func runModel(on buffer: CVPixelBuffer) throws -> [Float] {
        let provider: MLFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(
                dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)])
        } catch {
            throw FaceEmbeddingError.predictionFailed(error)
        }

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: provider)
        } catch {
            throw FaceEmbeddingError.predictionFailed(error)
        }

        guard let array = output.featureValue(for: outputName)?.multiArrayValue else {
            throw FaceEmbeddingError.outputMissing(outputName)
        }
        let vector = Self.toFloatArray(array)
        guard vector.count == expectedDimension else {
            throw FaceEmbeddingError.unexpectedOutputSize(vector.count)
        }
        return vector
    }

    /// Convert an MLMultiArray to [Float], handling the common element types.
    static func toFloatArray(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var out = [Float](repeating: 0, count: count)
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
            for i in 0..<count { out[i] = ptr[i] }
        case .double:
            let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            for i in 0..<count { out[i] = Float(ptr[i]) }
        case .float16:
            // No direct Float16 pointer read across all SDKs — go via NSNumber.
            for i in 0..<count { out[i] = array[i].floatValue }
        @unknown default:
            for i in 0..<count { out[i] = array[i].floatValue }
        }
        return out
    }
}
