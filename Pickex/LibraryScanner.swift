//
//  LibraryScanner.swift
//  Pickex
//
//  Step C: scan the photo library against a ReferenceProfile (Step B) and
//  stream back the assets that show the person.
//
//  Matching: a photo matches when ANY of its faces has
//  cosine(faceEmbedding, profile.embedding) >= matchThreshold. All faces are
//  checked (largest first, capped) because on group photos the target person
//  is usually not the largest face.
//
//  Threshold: `defaultMatchThreshold = 0.40`, measured on a real 18-photo
//  LFW test library against a 4-photo reference profile (production model):
//  same-person scores 0.651–0.779, other-people scores −0.033–0.078 — a
//  +0.57 gap, with 8/8 recall and 0/10 false positives anywhere in
//  0.30…0.50. 0.40 sits mid-gap; expose it in the config and tune on real
//  libraries (hard poses/occlusion pull positives down toward ~0.4).
//
//  Performance: images are decoded at `targetSize` (default 1024 px — plenty
//  for Vision + a 112×112 crop), a bounded TaskGroup keeps a few photos in
//  flight (Vision/Core ML saturate quickly; unbounded fan-out just burns
//  memory), and everything runs off the main thread. Cancellation: cancel the
//  consuming task / break out of the for-await loop and in-flight work stops.
//

import Photos
import CoreGraphics
import ImageIO
import Foundation

// MARK: - Results & configuration

public struct ScanMatch: Sendable {
    public let assetLocalIdentifier: String
    /// Best face score in the photo (cosine vs the reference embedding).
    public let similarity: Float
    /// Faces checked in this photo (context for the UI).
    public let faceCount: Int
}

public enum ScanEvent: Sendable {
    /// Emitted after every processed asset (drives a progress bar).
    case progress(processed: Int, total: Int)
    case match(ScanMatch)
}

public struct LibraryScannerConfig {
    /// Cosine threshold for "this face is the person" — see header note.
    public var matchThreshold: Float = LibraryScanner.defaultMatchThreshold
    /// Decode size for library photos (longest side, pixels).
    public var targetSize = CGSize(width: 1024, height: 1024)
    /// Max faces checked per photo (largest first).
    public var maxFacesPerPhoto: Int = 8
    /// Photos processed concurrently.
    public var concurrentPhotos: Int = 4
    public init() {}
}

// MARK: - Scanner

public final class LibraryScanner: @unchecked Sendable {

    public static let defaultMatchThreshold: Float = 0.40

    private let embedder: FaceEmbedder
    private let config: LibraryScannerConfig

    public init(embedder: FaceEmbedder, config: LibraryScannerConfig = LibraryScannerConfig()) {
        self.embedder = embedder
        self.config = config
    }

    /// Scan all image assets in the user's library against the profile.
    /// Consume with `for try await event in scanner.scanLibrary(against: profile)`.
    public func scanLibrary(against profile: ReferenceProfile,
                            fetchOptions: PHFetchOptions? = nil) -> AsyncThrowingStream<ScanEvent, Error> {
        let options = fetchOptions ?? {
            let o = PHFetchOptions()
            o.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            return o
        }()
        let assets = PHAsset.fetchAssets(with: .image, options: options)
        return scan(assets: assets, against: profile)
    }

    /// Scan an explicit fetch result (album, date range, …).
    public func scan(assets: PHFetchResult<PHAsset>,
                     against profile: ReferenceProfile) -> AsyncThrowingStream<ScanEvent, Error> {
        let reference = l2Normalize(profile.embedding)
        let total = assets.count
        let identifiers: [String] = (0..<total).map { assets.object(at: $0).localIdentifier }

        return AsyncThrowingStream { continuation in
            let worker = Task { [config, embedder] in
                var processed = 0
                var iterator = identifiers.makeIterator()

                try await withThrowingTaskGroup(of: ScanMatch?.self) { group in
                    var inFlight = 0

                    func addNext() -> Bool {
                        guard let id = iterator.next() else { return false }
                        group.addTask {
                            try Task.checkCancellation()
                            return Self.process(assetIdentifier: id,
                                                reference: reference,
                                                embedder: embedder,
                                                config: config)
                        }
                        inFlight += 1
                        return true
                    }

                    // Prime a bounded window, then keep it full.
                    while inFlight < max(1, config.concurrentPhotos), addNext() {}
                    while inFlight > 0 {
                        let match = try await group.next()!
                        inFlight -= 1
                        processed += 1
                        if let match { continuation.yield(.match(match)) }
                        continuation.yield(.progress(processed: processed, total: total))
                        _ = addNext()
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in worker.cancel() }
        }
    }

    // MARK: per-asset work (synchronous, runs inside a task-group child)

    private static func process(assetIdentifier: String,
                                reference: [Float],
                                embedder: FaceEmbedder,
                                config: LibraryScannerConfig) -> ScanMatch? {
        guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [assetIdentifier],
                                              options: nil).firstObject,
              let cgImage = requestCGImage(for: asset, targetSize: config.targetSize) else {
            return nil   // unloadable asset -> skip silently
        }

        let embeddings: [[Float]]
        do {
            embeddings = try embedder.embeddingsForAllFaces(in: cgImage,
                                                            maxFaces: config.maxFacesPerPhoto)
        } catch {
            return nil   // no face (normal) or per-photo failure -> no match
        }

        var best: Float = -1
        for vector in embeddings {
            best = max(best, dot(l2Normalize(vector), reference))
        }
        guard best >= config.matchThreshold else { return nil }
        return ScanMatch(assetLocalIdentifier: assetIdentifier,
                         similarity: best,
                         faceCount: embeddings.count)
    }

    /// Synchronous, downscaled decode via PHImageManager (hot path: no UIKit).
    private static func requestCGImage(for asset: PHAsset, targetSize: CGSize) -> CGImage? {
        let options = PHImageRequestOptions()
        options.isSynchronous = true            // we're already off-main in a child task
        options.deliveryMode = .highQualityFormat
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true   // iCloud originals

        var result: CGImage?
        PHImageManager.default().requestImage(for: asset,
                                              targetSize: targetSize,
                                              contentMode: .aspectFit,
                                              options: options) { image, _ in
            result = image?.cgImage
        }
        return result
    }
}
