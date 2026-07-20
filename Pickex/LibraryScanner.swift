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
    /// Highest face similarity seen so far — even below the match threshold.
    /// Makes "0 matches" debuggable: best=0.38 means "close, tune threshold",
    /// best=0.05 means "pipeline problem".
    public let bestSimilarity: Float
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

/// How a face embedding is scored against the reference profile.
public enum MatchingStrategy: Sendable {
    /// Cosine to the aggregated (mean) reference only. Best when the reference
    /// photos are all of a similar condition (the mean denoises them).
    case mean
    /// Max cosine to any single reference photo. Best when references span
    /// very different conditions the mean would blur.
    case bestOfSet
    /// max(mean, bestOfSet). Never scores a same-person photo lower than the
    /// mean, and rescues diverse-condition matches via individual references.
    /// Default — biased toward recall (what "mixed makeup/angle refs" needs).
    case combined
}

/// Precomputed reference vectors + strategy; scores a face embedding.
struct ReferenceMatcher: Sendable {
    let mean: [Float]          // L2-normalized aggregated reference
    let set: [[Float]]         // L2-normalized individual references
    let strategy: MatchingStrategy

    func score(_ embedding: [Float]) -> Float {
        let x = l2Normalize(embedding)
        let meanScore = { dot(x, mean) }
        let setScore = { set.map { dot(x, $0) }.max() }
        switch strategy {
        case .mean:      return meanScore()
        case .bestOfSet: return setScore() ?? meanScore()
        case .combined:  return max(meanScore(), setScore() ?? meanScore())
        }
    }
}

public struct LibraryScannerConfig {
    /// Cosine threshold for "this face is the person" — calibrated, see header.
    public var matchThreshold: Float = LibraryScanner.defaultMatchThreshold
    /// How faces are scored against the reference profile.
    public var matchingStrategy: MatchingStrategy = .combined
    /// Decode size for library photos (longest side, pixels). 1280 (not 640):
    /// group-photo faces are tiny fractions of the frame — at 640 they end up
    /// ~25-50px, and a 112px crop upscaled from that is mush. Mush embeddings
    /// cluster together and produced 0.8+ false matches. At 1280 the same
    /// faces carry enough pixels to embed honestly.
    public var targetSize = CGSize(width: 1280, height: 1280)
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
        let matcher = makeMatcher(profile)
        var results: [MatchResult] = []
        cache.forEachEntry { id, embeddings in
            var best: Float = -1
            for e in embeddings { best = max(best, matcher.score(e)) }
            if best >= config.matchThreshold {
                results.append(MatchResult(assetLocalIdentifier: id,
                                           similarity: best, matchedAt: Date()))
            }
        }
        return results.sorted { $0.similarity > $1.similarity }
    }

    private func makeMatcher(_ profile: ReferenceProfile) -> ReferenceMatcher {
        ReferenceMatcher(mean: l2Normalize(profile.embedding),
                         set: profile.referenceEmbeddings.map(l2Normalize),
                         strategy: config.matchingStrategy)
    }

    // MARK: Core scan

    public func scan(assets: PHFetchResult<PHAsset>,
                     against profile: ReferenceProfile) -> AsyncThrowingStream<ScanEvent, Error> {
        let matcher = makeMatcher(profile)
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
                var best: Float = -1
                var iterator = snapshot.makeIterator()

                func emitProgress(force: Bool = false) {
                    guard force || processed % max(1, config.progressGranularity) == 0 else { return }
                    continuation.yield(.progress(ScanProgress(
                        processed: processed, total: total,
                        matchCount: matchCount,
                        skippedNotLocal: notLocal,
                        servedFromCache: fromCache,
                        failedToProcess: failed,
                        bestSimilarity: best)))
                }

                do {
                    try await withThrowingTaskGroup(of: AssetOutcome.self) { group in
                        var inFlight = 0

                        func addNext() -> Bool {
                            guard let item = iterator.next() else { return false }
                            group.addTask {
                                try Task.checkCancellation()
                                return Self.process(item: item, matcher: matcher,
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
                                best = max(best, m.similarity)
                                if cached { fromCache += 1 }
                                continuation.yield(.match(m))
                            case .noMatch(let cached, let score):
                                best = max(best, score)
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

    /// DEBUG: load an asset exactly the way the scan does (same PHImageManager
    /// request + downscale), so the demo can render the scan-side crop and
    /// compare it to the reference-side crop.
    public func debugLoadImage(assetID: String) -> CGImage? {
        guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [assetID],
                                              options: nil).firstObject else { return nil }
        return Self.requestCGImage(for: asset, config: config).0
    }

    // MARK: per-asset work (inside a task-group child, off-main)

    private enum AssetOutcome: Sendable {
        case match(MatchResult, fromCache: Bool)
        case noMatch(fromCache: Bool, bestScore: Float)   // below threshold, score kept for diagnostics
        case skippedNotLocal
        case failed          // load/processing error this run; never cached
    }

    private static func process(item: (id: String, modified: Date?),
                                matcher: ReferenceMatcher,
                                embedder: FaceEmbedder,
                                cache: ScanCache?,
                                config: LibraryScannerConfig) -> AssetOutcome {
        // 1. Cache fast path — including cached "no faces" results.
        if let cached = cache?.lookup(localIdentifier: item.id, modificationDate: item.modified) {
            return outcome(for: cached, matcher: matcher,
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

        return outcome(for: embeddings, matcher: matcher,
                       id: item.id, threshold: config.matchThreshold, fromCache: false)
    }

    private static func outcome(for embeddings: [[Float]], matcher: ReferenceMatcher,
                                id: String, threshold: Float, fromCache: Bool) -> AssetOutcome {
        var best: Float = -1
        for e in embeddings { best = max(best, matcher.score(e)) }
        guard best >= threshold else { return .noMatch(fromCache: fromCache, bestScore: best) }
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
        func request(_ mode: PHImageRequestOptionsDeliveryMode,
                     network: Bool) -> (CGImage?, Bool) {
            let options = PHImageRequestOptions()
            options.isSynchronous = true        // we're already in a worker task
            options.deliveryMode = mode
            options.resizeMode = .fast
            options.isNetworkAccessAllowed = network

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

        // 1. LOCAL first, even for iCloud-only assets: with "optimize storage"
        //    the device already holds a screen-size preview of nearly every
        //    photo, and that is plenty for detection + a 112px crop. Only a
        //    too-small preview (tiny thumbnail) is rejected — recognition on
        //    it would silently underperform.
        var (image, inCloud) = request(.highQualityFormat, network: false)
        if image == nil {
            let (fast, fastCloud) = request(.fastFormat, network: false)
            if let fast, max(fast.width, fast.height) >= 700 { image = fast }
            inCloud = inCloud || fastCloud
        }
        // 2. Only when nothing usable exists locally: hit the network (when
        //    allowed), but with a hard timeout so one slow/stalled download can
        //    never block its worker — and thus the whole scan — indefinitely.
        //    A photo that times out stays uncached and is retried next scan.
        if image == nil && config.allowNetworkAccess {
            (image, inCloud) = requestWithTimeout(for: asset, config: config, seconds: 10)
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

    /// Network image fetch with a hard timeout. Uses an ASYNCHRONOUS request
    /// (its completion runs on PHImageManager's own queue, not the Swift
    /// cooperative pool) and blocks the worker on a semaphore only until the
    /// result arrives or `seconds` elapse — then cancels. This is what keeps a
    /// stalled iCloud download from starving the concurrency pool and freezing
    /// the whole scan (the failure mode of synchronous network requests).
    private static func requestWithTimeout(for asset: PHAsset,
                                           config: LibraryScannerConfig,
                                           seconds: Double) -> (CGImage?, Bool) {
        let options = PHImageRequestOptions()
        options.isSynchronous = false
        options.deliveryMode = .fastFormat        // single delivery, resized derivative
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true

        let sem = DispatchSemaphore(value: 0)
        var image: CGImage?
        var inCloud = false
        var settled = false
        let id = PHImageManager.default().requestImage(
            for: asset, targetSize: config.targetSize,
            contentMode: .aspectFit, options: options) { ui, info in
            let degraded = (info?[PHImageResultIsDegradedKey] as? NSNumber)?.boolValue ?? false
            if let cg = ui?.cgImage, !degraded { image = cg }
            inCloud = (info?[PHImageResultIsInCloudKey] as? NSNumber)?.boolValue ?? false
            if !degraded && !settled { settled = true; sem.signal() }  // final delivery or error
        }
        if sem.wait(timeout: .now() + seconds) == .timedOut {
            PHImageManager.default().cancelImageRequest(id)
            return (nil, true)
        }
        return (image, image == nil && inCloud)
    }
}
