//
//  ReferenceProfileBuilder+PHPicker.swift
//  Pickex
//
//  Thin PhotosUI convenience over the decoupled `build(from: [ReferenceImage])`
//  core, so the reference-picker screen can pass PHPickerResults directly.
//

#if canImport(PhotosUI) && canImport(UIKit)
import PhotosUI
import UIKit

extension ReferenceProfileBuilder {

    /// Load the picked photos and build the profile.
    ///
    /// Note: a PHPickerResult whose image simply can't be loaded (rare) is
    /// dropped and does NOT appear in `skipped` — `skipped` is reserved for
    /// photos that were processed but had no usable face. If every result
    /// fails to load, `build(from:)` throws `.noPhotosProvided`.
    public func build(from results: [PHPickerResult]) async throws -> ReferenceProfile {
        var images: [ReferenceImage] = []
        for result in results {
            if let image = try? await Self.loadReferenceImage(from: result.itemProvider) {
                images.append(image)
            }
        }
        return try await build(from: images)
    }

    private static func loadReferenceImage(from provider: NSItemProvider) async throws -> ReferenceImage? {
        guard provider.canLoadObject(ofClass: UIImage.self) else { return nil }
        return try await withCheckedThrowingContinuation { continuation in
            provider.loadObject(ofClass: UIImage.self) { object, error in
                if let error = error { continuation.resume(throwing: error); return }
                guard let uiImage = object as? UIImage, let cgImage = uiImage.cgImage else {
                    continuation.resume(returning: nil); return
                }
                let orientation = CGImagePropertyOrientation(uiImage.imageOrientation)
                continuation.resume(returning: ReferenceImage(cgImage: cgImage, orientation: orientation))
            }
        }
    }
}

extension CGImagePropertyOrientation {
    /// Bridge UIKit's imageOrientation to the CGImagePropertyOrientation that
    /// Vision / FacePreprocessor expect.
    init(_ orientation: UIImage.Orientation) {
        switch orientation {
        case .up: self = .up
        case .upMirrored: self = .upMirrored
        case .down: self = .down
        case .downMirrored: self = .downMirrored
        case .left: self = .left
        case .leftMirrored: self = .leftMirrored
        case .right: self = .right
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}
#endif
