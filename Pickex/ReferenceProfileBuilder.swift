//
//  ReferenceProfileBuilder.swift
//  Pickex
//
//  Step B: turn the user's 1–4 reference photos into ONE reference embedding
//  ("this person") that the whole library is later matched against.
//
//  Reuses FaceEmbedder (Step A/`FaceEmbedder.swift`) for the per-image
//  embedding and FacePreprocessError for the "no face" case. The result type
//  `ReferenceProfile` is consumed by the reference-picker UI and by the library
//  scan (Step C), so it's intentionally explicit and stable.
//

import CoreGraphics
import ImageIO
import Foundation

// MARK: - Result types (consumed by UI + scan step)

public struct ReferenceProfile {
    /// Final 512-d reference vector, L2-normalized (cosine matching downstream).
    public let embedding: [Float]
    /// The individual L2-normalized per-photo embeddings (the "set"). Lets the
    /// scanner match against the best-fitting reference instead of only the
    /// mean — crucial when references span different conditions (e.g. makeup
    /// vs. no-makeup, different angles) that a single mean would blur together.
    public let referenceEmbeddings: [[Float]]
    /// How many photos yielded a usable face.
    public let usedPhotoCount: Int
    /// Photos that were skipped, with the reason (surface these in the UI).
    public let skipped: [SkippedPhoto]
    /// Indices of photos where >1 face was detected — UI may ask
    /// "multiple faces detected, did we pick the right one?".
    public let multipleFaceWarnings: [Int]
    /// Set when two reference photos disagree too much (possible wrong person).
    /// A soft warning, not a failure — the user decides.
    public let consistencyWarning: ConsistencyWarning?
}

public struct SkippedPhoto: Equatable {
    public let index: Int          // index into the input photo array
    public let reason: SkipReason
    public init(index: Int, reason: SkipReason) { self.index = index; self.reason = reason }
}

public enum SkipReason: Equatable, Sendable {
    case noFaceDetected
    case processingError(String)   // String (not Error) -> Equatable / UI-friendly
}

public struct SimilarityPair: Equatable {
    public let photoA: Int         // input indices
    public let photoB: Int
    public let similarity: Float
}

public struct ConsistencyWarning {
    public let minSimilarity: Float
    public let flaggedPairs: [SimilarityPair]   // pairs below the threshold
    public let message: String
}

public enum ReferenceProfileError: Error, Equatable {
    case noPhotosProvided
    /// No face in ANY photo -> UI: "Kein Gesicht erkannt, bitte andere Fotos wählen".
    /// Carries the per-photo skip reasons so the UI can show WHY (a photo with
    /// no detectable face vs. a processing/model error look identical otherwise).
    case noFaceInAnyPhoto(skipped: [SkippedPhoto])
}

/// One reference photo, decoupled from PhotosUI so the builder is unit-testable.
/// (CGImage is immutable and thread-safe.)
public struct ReferenceImage: @unchecked Sendable {
    public let cgImage: CGImage
    public let orientation: CGImagePropertyOrientation
    public init(cgImage: CGImage, orientation: CGImagePropertyOrientation = .up) {
        self.cgImage = cgImage
        self.orientation = orientation
    }
}

// MARK: - Aggregation strategy (swappable)

public protocol EmbeddingAggregator: Sendable {
    /// Combine raw per-photo embeddings into one final vector.
    func aggregate(_ embeddings: [[Float]]) -> [Float]
}

/// Default: L2-normalize each embedding, average, then L2-normalize the mean.
///
/// Why mean+normalize is the pragmatic default: it's cheap, and averaging a few
/// shots of the same person cancels per-photo pose/lighting noise into a stabler
/// template than any single shot. Per-embedding normalization first prevents one
/// high-magnitude embedding from dominating the average. Because it's behind the
/// `EmbeddingAggregator` protocol, we can later swap in e.g. "keep the set and
/// take the best match at query time" without touching callers.
public struct MeanAggregator: EmbeddingAggregator {
    public init() {}
    public func aggregate(_ embeddings: [[Float]]) -> [Float] {
        guard let dim = embeddings.first?.count, dim > 0, !embeddings.isEmpty else { return [] }
        var sum = [Float](repeating: 0, count: dim)
        for e in embeddings {
            let u = l2Normalize(e)
            for i in 0..<dim { sum[i] += u[i] }
        }
        let mean = sum.map { $0 / Float(embeddings.count) }
        return l2Normalize(mean)
    }
}

// MARK: - Builder

public final class ReferenceProfileBuilder {

    /// Pairwise cosine below this ⇒ consistency warning. Chosen from the Step-A
    /// discrimination data: same person measured ~0.72, different people ~0.0–0.07,
    /// so 0.4 separates cleanly with margin. Exposed as a constant to tune later.
    public static let defaultConsistencyThreshold: Float = 0.4

    private let embedder: FaceEmbedder
    private let aggregator: EmbeddingAggregator
    private let consistencyThreshold: Float

    public init(embedder: FaceEmbedder,
                aggregator: EmbeddingAggregator = MeanAggregator(),
                consistencyThreshold: Float = ReferenceProfileBuilder.defaultConsistencyThreshold) {
        self.embedder = embedder
        self.aggregator = aggregator
        self.consistencyThreshold = consistencyThreshold
    }

    /// Core API — the 1–4 photos are embedded in parallel (TaskGroup) so the
    /// user doesn't wait serially right after picking.
    public func build(from images: [ReferenceImage]) async throws -> ReferenceProfile {
        guard !images.isEmpty else { throw ReferenceProfileError.noPhotosProvided }

        // 1. Embed every photo concurrently; never throw out of a child task —
        //    a missing face is a skip, not a failure.
        var outcomes = [PhotoOutcome?](repeating: nil, count: images.count)
        await withTaskGroup(of: PhotoOutcome.self) { group in
            for (index, image) in images.enumerated() {
                group.addTask { [embedder] in
                    do {
                        let r = try embedder.embedding(from: image.cgImage,
                                                       orientation: image.orientation)
                        return .success(index: index, vector: r.vector, faceCount: r.faceCount)
                    } catch FacePreprocessError.noFaceFound {
                        return .skipped(index: index, reason: .noFaceDetected)
                    } catch {
                        return .skipped(index: index, reason: .processingError(String(describing: error)))
                    }
                }
            }
            for await outcome in group { outcomes[outcome.index] = outcome }
        }

        // 2. Split into used / skipped (kept in input order).
        var used: [(index: Int, vector: [Float])] = []
        var skipped: [SkippedPhoto] = []
        var multipleFaceWarnings: [Int] = []
        for outcome in outcomes.compactMap({ $0 }) {
            switch outcome {
            case let .success(index, vector, faceCount):
                used.append((index, vector))
                if faceCount > 1 { multipleFaceWarnings.append(index) }
            case let .skipped(index, reason):
                skipped.append(SkippedPhoto(index: index, reason: reason))
            }
        }
        used.sort { $0.index < $1.index }
        multipleFaceWarnings.sort()

        // 3. Complete failure: not a single face anywhere.
        guard !used.isEmpty else {
            throw ReferenceProfileError.noFaceInAnyPhoto(skipped: skipped)
        }

        // 4. Consistency check (soft warning) across the valid embeddings.
        let warning = consistencyWarning(for: used)

        // 5. Aggregate -> final reference embedding, and keep the individual
        //    normalized embeddings for best-of-set matching.
        let embedding = aggregator.aggregate(used.map { $0.vector })
        let referenceEmbeddings = used.map { l2Normalize($0.vector) }

        return ReferenceProfile(embedding: embedding,
                                referenceEmbeddings: referenceEmbeddings,
                                usedPhotoCount: used.count,
                                skipped: skipped,
                                multipleFaceWarnings: multipleFaceWarnings,
                                consistencyWarning: warning)
    }

    // MARK: helpers

    private func consistencyWarning(for used: [(index: Int, vector: [Float])]) -> ConsistencyWarning? {
        guard used.count >= 2 else { return nil }
        let normed = used.map { l2Normalize($0.vector) }
        var flagged: [SimilarityPair] = []
        var minSim: Float = 1.0
        for i in 0..<normed.count {
            for j in (i + 1)..<normed.count {
                let s = dot(normed[i], normed[j])
                minSim = min(minSim, s)
                if s < consistencyThreshold {
                    flagged.append(SimilarityPair(photoA: used[i].index,
                                                  photoB: used[j].index,
                                                  similarity: s))
                }
            }
        }
        guard !flagged.isEmpty else { return nil }
        let msg = "Ein oder mehrere Referenzfotos weichen stark voneinander ab "
            + "(geringste Ähnlichkeit \(String(format: "%.2f", minSim))). "
            + "Möglicherweise zeigt ein Foto eine andere Person."
        return ConsistencyWarning(minSimilarity: minSim, flaggedPairs: flagged, message: msg)
    }

    private enum PhotoOutcome: Sendable {
        case success(index: Int, vector: [Float], faceCount: Int)
        case skipped(index: Int, reason: SkipReason)
        var index: Int {
            switch self {
            case let .success(i, _, _): return i
            case let .skipped(i, _): return i
            }
        }
    }
}

// MARK: - vector math

func l2Normalize(_ v: [Float]) -> [Float] {
    let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
    return norm > 0 ? v.map { $0 / norm } : v
}

func dot(_ a: [Float], _ b: [Float]) -> Float {
    var s: Float = 0
    for i in 0..<min(a.count, b.count) { s += a[i] * b[i] }
    return s
}
