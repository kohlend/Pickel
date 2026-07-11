//
//  LibraryScanner.swift
//  Pickex
//
//  Step C — the core of the app: scan the ENTIRE photo library (10k+ photos)
//  against a ReferenceProfile (Step B), collect matches with confidence
//  scores, report progress, support cancellation, and never recompute photos
//  that are already in the ScanCache.
//
//  ── Design decisions (each with the "why") ──────────────────────────────────
//  • Scope: `PHAsset.fetchAssets(with: .image)` — still images only. Videos
//    are excluded BY THE FETCH, not silently later. Live Photos are .image
//    assets; PHImageManager returns their still key frame, which is what we
//    scan.
//  • iCloud: `isNetworkAccessAllowed` is OFF by default — a library scan must
//    never surprise-download thousands of originals over cellular. Assets
//    whose pixels aren't on device are counted in `ScanProgress.skippedNotLocal`
//    so the UI can say "X photos couldn't be checked (not on this device)".
//    Flip `config.allowNetworkAccess` for an opt-in "deep scan on Wi-Fi".
//  • Decode size 640 px (longest side), `.fastFormat`, `.fast` resize: 640 px
//    is plenty for Vision face detection + a 112×112 aligned crop, decodes
//    ~3-4× faster than 1024+, and fastFormat lets Photos hand us an existing
//    thumbnail instead of decoding the original.
//  • Concurrency = batching: a bounded TaskGroup keeps exactly
//    `concurrentPhotos` (default 4) images in flight — that IS the memory
//    cap (≈4 × ~1.5 MB decoded + model state), unlike loading 50-photo
//    batches into arrays. Vision/Core ML saturate a phone well below 4-way
//    parallelism, so more tasks would only burn RAM. Each child task wraps
//    its work in `autoreleasepool` (PHImageManager returns autoreleased
//    UIImages).
//  • Matching: EVERY face in the photo is embedded (largest first, capped at
//    `maxFacesPerPhoto`) — on group photos the target person is usually not
//    the largest face. Photo confidence = max cosine over its faces.
//  • Threshold 0.40 (`defaultMatchThreshold`): calibrated on a real LFW test
//    library (README §Step-C): positives 0.651–0.779, negatives ≤ 0.078.
//    (An earlier draft suggested ~0.70 — that would sit INSIDE the measured
//    positive range and drop half the true matches.) Config knob for tuning.
//  • Progress is throttled: `.progress` every `progressGranularity` assets
//    (default 20) plus a final one — not per asset (UI-update flood).
//  • Cancellation: end the `for try await` loop (or cancel the consuming
//    Task). Matches already delivered stay with the caller, and everything
//    computed so far is already in the ScanCache — an aborted scan loses no
//    work; the next scan resumes from cache.
//  • Cache: raw per-face embeddings per localIdentifier+modificationDate
//    (see ScanCache.swift), INCLUDING "0 faces" results — most photos have no
//    faces, and skipping them on re-scan is where the speedup comes from.
//    `rematchFromCache(against:)` matches a NEW reference profile against all
//    cached embeddings without touching the library at all.
//

import Photos
import CoreGraphics
import ImageIO
import Foundation
import UIKit

// MARK: - Public types

public struct ScanProgress: Sendable {
    public let processed: Int
    public let total: Int
    public let matchCount: Int
    /// iCloud-only assets skipped because network access is disabled.
    public let skippedNotLocal: Int
    /// How many processed assets were answered from the ScanCache.
    public let servedFromCache: Int
    /// Assets that could not be loaded/processed this run (never cached).
    public let failedToProcess: Int
    public var fraction: Double { total > 0 ? Double(processed) / Double(total) : 1 }
}

public struct MatchResult: Sendable, Codable {
    public let assetLocalIdentifier: String
    /// Best cosine similarity over all faces in the photo.
    public let similarity: Float
    public let matchedAt: Date
}

public enum ScanEvent: Sendable {
    /// Throttled (every `progressGranularity` assets, and once at the end).
    case progress(ScanProgress)
    /// Emitted immediately when an asset matches.
    case match(MatchResult)
}

public struct LibraryScannerConfig {
    /// Cosine threshold for "this face is the person" — calibrated, see header.
    public var matchThreshold: Float = LibraryScanner.defaultMatchThreshold
    /// Decode size for library photos (longest side, pixels).
    public var targetSize = CGSize(width: 640, height: 640)
    /// Max faces embedded per photo (largest first).
    public var maxFacesPerPhoto: Int = 8
    /// Photos in flight at once — this bounds peak memory.
    public var concurrentPhotos: Int = 4
    /// Allow downloading iCloud-only originals. OFF by default (see header).
    public var allowNetworkAccess: Bool = false
    /// Emit `.progress` every N processed assets.
    public var progressGranularity: Int = 20
    public init() {}
}

// MARK: - Scanner

public final class LibraryScanner: @unchecked Sendable {

    public static let defaultMatchThreshold: Float = 0.40

    private let embedder: FaceEmbedder
    private let cache: ScanCache?
    private let config: LibraryScannerConfig

    /// `cache: nil` disables persistence (every scan recomputes everything).
    public init(embedder: FaceEmbedder,
                cache: ScanCache?,
                config: LibraryScannerConfig = LibraryScannerConfig()) {
        self.embedder = embedder
        self.cache = cache
        self.config = config
    }

    // MARK: Primary API — streaming

    /// Scan every image in the library. Consume with
    /// `for try await event in scanner.scanEvents(against: profile) { … }`.
    public func scanEvents(against profile: ReferenceProfile,
                           fetchOptions: PHFetchOptions? = nil) -> AsyncThrowingStream<ScanEvent, Error> {
        let options = fetchOptions ?? {
            let o = PHFetchOptions()
            o.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            return o
        }()
        // .image excludes videos by construction; Live Photos are images here.
        let assets = PHAsset.fetchAssets(with: .image, options: options)
        return scan(assets: assets, against: profile)
    }

    /// Convenience with the collect-style signature: runs the stream to
    /// completion, forwards throttled progress, returns matches sorted by
    /// similarity (best first).
    public func scan(reference: ReferenceProfile,
                     progress: @escaping @Sendable (ScanProgress) -> Void) async throws -> [MatchResult] {
        var matches: [MatchResult] = []
        for try await event in scanEvents(against: reference) {
            switch event {
            case .progress(let p): progress(p)
            case .match(let m): matches.append(m)
            }
        }
        return matches.sorted { $0.similarity > $1.similarity }
    }

    /// Match a NEW reference profile purely against cached embeddings —
    /// milliseconds instead of a full scan. Only covers photos that are in
    /// the cache; run a normal scan afterwards to pick up new photos.
    public func rematchFromCache(against profile: ReferenceProfile) -> [MatchResult] {
        guard let cache else { return [] }
        let reference = l2Normalize(profile.embedding)
        var results: [MatchResult] = []
        cache.forEachEntry { id, embeddings in
            var best: Float = -1
            for e in embeddings { best = max(best, dot(l2Normalize(e), reference)) }
            if best >= config.matchThreshold {
                results.append(MatchResult(assetLocalIdentifier: id,
                                           similarity: best, matchedAt: Date()))
            }
        }
        return results.sorted { $0.similarity > $1.similarity }
    }

    // MARK: Core scan

    public func scan(assets: PHFetchResult<PHAsset>,
                     against profile: ReferenceProfile) -> AsyncThrowingStream<ScanEvent, Error> {
        let reference = l2Normalize(profile.embedding)
        let total = assets.count
        // Snapshot (id, modificationDate) up front — cheap metadata, avoids
        // holding PHAssets across tasks.
        var snapshot: [(id: String, modified: Date?)] = []
        snapshot.reserveCapacity(total)
        assets.enumerateObjects { asset, _, _ in
            snapshot.append((asset.localIdentifier, asset.modificationDate))
        }

        return AsyncThrowingStream { continuation in
            let worker = Task { [config, embedder, cache] in
                // Drop cache rows for photos that no longer exist.
                cache?.prune(keeping: Set(snapshot.map(\.id)))

                var processed = 0, matchCount = 0, notLocal = 0, fromCache = 0, failed = 0
                var iterator = snapshot.makeIterator()

                func emitProgress(force: Bool = false) {
                    guard force || processed % max(1, config.progressGranularity) == 0 else { return }
                    continuation.yield(.progress(ScanProgress(
                        processed: processed, total: total,
                        matchCount: matchCount,
                        skippedNotLocal: notLocal,
                        servedFromCache: fromCache,
                        failedToProcess: failed)))
                }

                do {
                    try await withThrowingTaskGroup(of: AssetOutcome.self) { group in
                        var inFlight = 0

                        func addNext() -> Bool {
                            guard let item = iterator.next() else { return false }
                            group.addTask {
                                try Task.checkCancellation()
                                return Self.process(item: item, reference: reference,
                                                    embedder: embedder, cache: cache,
                                                    config: config)
                            }
                            inFlight += 1
                            return true
                        }

                        // Bounded window: exactly `concurrentPhotos` in flight.
                        while inFlight < max(1, config.concurrentPhotos), addNext() {}
                        while inFlight > 0 {
                            let outcome = try await group.next()!
                            inFlight -= 1
                            processed += 1
                            switch outcome {
                            case .match(let m, let cached):
                                matchCount += 1
                                if cached { fromCache += 1 }
                                continuation.yield(.match(m))
                            case .noMatch(let cached):
                                if cached { fromCache += 1 }
                            case .skippedNotLocal:
                                notLocal += 1
                            case .failed:
                                failed += 1
                            }
                            emitProgress()
                            _ = addNext()
                        }
                    }
                    emitProgress(force: true)
                    continuation.finish()
                } catch {
                    // Cancellation or a hard failure: matches already yielded
                    // stay with the caller; cache already holds computed work.
                    emitProgress(force: true)
                    continuation.finish(throwing: error is CancellationError ? nil : error)
                }
            }
            continuation.onTermination = { _ in worker.cancel() }
        }
    }

    // MARK: per-asset work (inside a task-group child, off-main)

    private enum AssetOutcome: Sendable {
        case match(MatchResult, fromCache: Bool)
        case noMatch(fromCache: Bool)
        case skippedNotLocal
        case failed          // load/processing error this run; never cached
    }

    private static func process(item: (id: String, modified: Date?),
                                reference: [Float],
                                embedder: FaceEmbedder,
                                cache: ScanCache?,
                                config: LibraryScannerConfig) -> AssetOutcome {
        // 1. Cache fast path — including cached "no faces" results.
        if let cached = cache?.lookup(localIdentifier: item.id, modificationDate: item.modified) {
            return outcome(for: cached, reference: reference,
                           id: item.id, threshold: config.matchThreshold, fromCache: true)
        }

        // 2. Load a downscaled decode; detect iCloud-only assets.
        var embeddings: [[Float]] = []
        var notLocal = false
        var computeFailed = false
        autoreleasepool {
            guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [item.id],
                                                  options: nil).firstObject else { return }
            let (cgImage, inCloud) = requestCGImage(for: asset, config: config)
            guard let cgImage else {
                // Cloud-only -> count as skipped; any other load failure is
                // transient and must not be cached as "0 faces".
                if inCloud { notLocal = true } else { computeFailed = true }
                return
            }
            do {
                embeddings = try embedder.embeddingsForAllFaces(
                    in: cgImage, maxFaces: config.maxFacesPerPhoto)
            } catch FacePreprocessError.noFaceFound {
                embeddings = []          // genuine "no faces" — cacheable result
            } catch {
                computeFailed = true     // transient error — must NOT be cached,
                                         // or a broken run poisons every re-scan
            }
        }
        if notLocal { return .skippedNotLocal }
        if computeFailed { return .failed }

        // 3. Persist raw embeddings (also the empty "no faces" result).
        cache?.store(localIdentifier: item.id, modificationDate: item.modified,
                     embeddings: embeddings)

        return outcome(for: embeddings, reference: reference,
                       id: item.id, threshold: config.matchThreshold, fromCache: false)
    }

    private static func outcome(for embeddings: [[Float]], reference: [Float],
                                id: String, threshold: Float, fromCache: Bool) -> AssetOutcome {
        var best: Float = -1
        for e in embeddings { best = max(best, dot(l2Normalize(e), reference)) }
        guard best >= threshold else { return .noMatch(fromCache: fromCache) }
        return .match(MatchResult(assetLocalIdentifier: id,
                                  similarity: best, matchedAt: Date()),
                      fromCache: fromCache)
    }

    /// Downscaled decode via PHImageManager. Returns (image, isCloudOnly).
    /// `.fastFormat` first (Photos may serve an existing thumbnail — exactly
    /// what a mass scan wants); assets without thumbnail resources fail that
    /// with PHPhotosError 3303 "No resource found matching image request
    /// spec" (e.g. photos imported into the simulator), so fall back to
    /// `.highQualityFormat` before giving up.
    private static func requestCGImage(for asset: PHAsset,
                                       config: LibraryScannerConfig) -> (CGImage?, Bool) {
        func request(_ mode: PHImageRequestOptionsDeliveryMode) -> (CGImage?, Bool) {
            let options = PHImageRequestOptions()
            options.isSynchronous = true        // we're already in a worker task
            options.deliveryMode = mode
            options.resizeMode = .fast
            options.isNetworkAccessAllowed = config.allowNetworkAccess

            var image: CGImage?
            var inCloud = false
            PHImageManager.default().requestImage(for: asset,
                                                  targetSize: config.targetSize,
                                                  contentMode: .aspectFit,
                                                  options: options) { ui, info in
                image = ui?.cgImage
                inCloud = (info?[PHImageResultIsInCloudKey] as? NSNumber)?.boolValue ?? false
            }
            return (image, inCloud)
        }

        var (image, inCloud) = request(.fastFormat)
        if image == nil && !inCloud {
            (image, inCloud) = request(.highQualityFormat)
        }
        // Last resort: fetch the ORIGINAL data and downsample it ourselves via
        // ImageIO. Bypasses the Photos thumbnail pipeline entirely — some
        // assets (e.g. photos imported into the simulator) fail both request
        // modes above with PHPhotosError 3303.
        if image == nil && !inCloud {
            let dataOptions = PHImageRequestOptions()
            dataOptions.isSynchronous = true
            dataOptions.isNetworkAccessAllowed = config.allowNetworkAccess
            var data: Data?
            PHImageManager.default().requestImageDataAndOrientation(for: asset,
                                                                    options: dataOptions) { d, _, _, info in
                data = d
                inCloud = (info?[PHImageResultIsInCloudKey] as? NSNumber)?.boolValue ?? false
            }
            if let data,
               let source = CGImageSourceCreateWithData(data as CFData,
                                                        [kCGImageSourceShouldCache: false] as CFDictionary) {
                let side = Int(max(config.targetSize.width, config.targetSize.height))
                let thumbOptions: [CFString: Any] = [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,   // applies EXIF orientation
                    kCGImageSourceShouldCacheImmediately: true,
                    kCGImageSourceThumbnailMaxPixelSize: side,
                ]
                image = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOptions as CFDictionary)
            }
        }
        return (image, image == nil && inCloud)
    }
}
