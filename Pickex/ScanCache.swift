//
//  ScanCache.swift
//  Pickex
//
//  Persistent per-asset embedding cache for the library scan (Step C), so a
//  second scan doesn't recompute Vision + Core ML for photos it has already
//  seen — and so a NEW reference profile can be re-matched against cached
//  embeddings in milliseconds without touching the photo library at all.
//
//  Storage: built-in SQLite3 (no dependency), WAL mode. One table:
//
//      assets(
//        local_id     TEXT PRIMARY KEY,   -- PHAsset.localIdentifier
//        modified_at  REAL,               -- PHAsset.modificationDate (epoch), 0 if nil
//        face_count   INTEGER,            -- 0 = scanned and no faces (cache that too!)
//        embeddings   BLOB                -- face_count × 512 × Float32, concatenated
//      )
//
//  Why SQLite over GRDB/JSON: GRDB would add a dependency for a one-table DB;
//  a Codable file would be rewritten wholesale on every batch (~25 MB at 10k
//  photos). SQLite writes incrementally and is crash-safe with WAL.
//
//  Invalidation:
//  • Edited photo → modificationDate changes → lookup() misses → recomputed
//    and overwritten (INSERT OR REPLACE).
//  • Deleted photo → prune(keeping:) at scan start removes orphaned rows.
//  • Raw embeddings (not match results) are stored, so the cache is
//    reference-profile-independent by design.
//
//  Size: ~2 KB per average photo (≈1 face) → ~25 MB for a 10k library.
//

import Foundation
import SQLite3

public enum ScanCacheError: Error {
    case openFailed(String)
    case sqlFailed(String)
}

public final class ScanCache: @unchecked Sendable {

    public static let embeddingDimension = 512

    private var db: OpaquePointer?
    // All DB access is funneled through one serial queue: sqlite handles are
    // not thread-safe per-connection, and scan write volume (a few rows/sec)
    // doesn't justify a pool.
    private let queue = DispatchQueue(label: "pickex.scancache")

    /// Default location: Application Support/Pickex/scancache.sqlite
    public convenience init() throws {
        let dir = try FileManager.default.url(for: .applicationSupportDirectory,
                                              in: .userDomainMask,
                                              appropriateFor: nil, create: true)
            .appendingPathComponent("Pickex", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try self.init(url: dir.appendingPathComponent("scancache.sqlite"))
    }

    public init(url: URL) throws {
        var handle: OpaquePointer?
        guard sqlite3_open(url.path, &handle) == SQLITE_OK, let handle else {
            let msg = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            sqlite3_close(handle)
            throw ScanCacheError.openFailed(msg)
        }
        db = handle
        try exec("PRAGMA journal_mode=WAL;")
        try exec("""
            CREATE TABLE IF NOT EXISTS assets(
              local_id    TEXT PRIMARY KEY,
              modified_at REAL NOT NULL,
              face_count  INTEGER NOT NULL,
              embeddings  BLOB
            );
            """)
    }

    deinit { sqlite3_close(db) }

    // MARK: - API

    /// Cached embeddings for the asset, or nil when absent OR stale
    /// (modificationDate mismatch). An empty array is a valid hit:
    /// "scanned before, no faces".
    public func lookup(localIdentifier: String, modificationDate: Date?) -> [[Float]]? {
        queue.sync {
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db,
                "SELECT modified_at, face_count, embeddings FROM assets WHERE local_id = ?;",
                -1, &stmt, nil) == SQLITE_OK else { return nil }
            sqlite3_bind_text(stmt, 1, localIdentifier, -1, Self.transient)
            guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }

            let cachedMod = sqlite3_column_double(stmt, 0)
            guard cachedMod == Self.epoch(modificationDate) else { return nil }  // stale

            let faceCount = Int(sqlite3_column_int(stmt, 1))
            guard faceCount > 0 else { return [] }
            guard let blob = sqlite3_column_blob(stmt, 2) else { return nil }
            let byteCount = Int(sqlite3_column_bytes(stmt, 2))
            let expected = faceCount * Self.embeddingDimension * MemoryLayout<Float32>.size
            guard byteCount == expected else { return nil }  // corrupt row -> treat as miss

            let flat = [Float32](UnsafeBufferPointer(
                start: blob.assumingMemoryBound(to: Float32.self),
                count: faceCount * Self.embeddingDimension))
            return (0..<faceCount).map { f in
                Array(flat[f * Self.embeddingDimension ..< (f + 1) * Self.embeddingDimension])
            }
        }
    }

    /// Store (or overwrite) the scan result for an asset. Pass an empty array
    /// for "no faces found" — caching that is what makes re-scans fast, since
    /// most library photos have no faces.
    public func store(localIdentifier: String, modificationDate: Date?, embeddings: [[Float]]) {
        queue.sync {
            var flat = [Float32]()
            flat.reserveCapacity(embeddings.count * Self.embeddingDimension)
            for e in embeddings { flat.append(contentsOf: e) }

            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db,
                "INSERT OR REPLACE INTO assets(local_id, modified_at, face_count, embeddings) VALUES (?,?,?,?);",
                -1, &stmt, nil) == SQLITE_OK else { return }
            sqlite3_bind_text(stmt, 1, localIdentifier, -1, Self.transient)
            sqlite3_bind_double(stmt, 2, Self.epoch(modificationDate))
            sqlite3_bind_int(stmt, 3, Int32(embeddings.count))
            if flat.isEmpty {
                sqlite3_bind_null(stmt, 4)
            } else {
                flat.withUnsafeBytes { raw in
                    _ = sqlite3_bind_blob(stmt, 4, raw.baseAddress, Int32(raw.count), Self.transient)
                }
            }
            _ = sqlite3_step(stmt)
        }
    }

    /// Remove rows for photos that no longer exist in the library.
    /// Call once at scan start with the current identifier set.
    @discardableResult
    public func prune(keeping identifiers: Set<String>) -> Int {
        queue.sync {
            var removed = 0
            var toDelete: [String] = []
            var stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "SELECT local_id FROM assets;", -1, &stmt, nil) == SQLITE_OK {
                while sqlite3_step(stmt) == SQLITE_ROW {
                    if let c = sqlite3_column_text(stmt, 0) {
                        let id = String(cString: c)
                        if !identifiers.contains(id) { toDelete.append(id) }
                    }
                }
            }
            sqlite3_finalize(stmt)
            for id in toDelete {
                var del: OpaquePointer?
                if sqlite3_prepare_v2(db, "DELETE FROM assets WHERE local_id = ?;", -1, &del, nil) == SQLITE_OK {
                    sqlite3_bind_text(del, 1, id, -1, Self.transient)
                    if sqlite3_step(del) == SQLITE_DONE { removed += 1 }
                }
                sqlite3_finalize(del)
            }
            return removed
        }
    }

    /// Iterate every cached asset (id + embeddings). Powers "new reference
    /// profile → re-match instantly without re-scanning the library".
    public func forEachEntry(_ body: (String, [[Float]]) -> Void) {
        // Collect under the queue, call out without holding it.
        let rows: [(String, [[Float]])] = queue.sync {
            var out: [(String, [[Float]])] = []
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db,
                "SELECT local_id, face_count, embeddings FROM assets WHERE face_count > 0;",
                -1, &stmt, nil) == SQLITE_OK else { return out }
            while sqlite3_step(stmt) == SQLITE_ROW {
                guard let c = sqlite3_column_text(stmt, 0) else { continue }
                let id = String(cString: c)
                let n = Int(sqlite3_column_int(stmt, 1))
                guard n > 0, let blob = sqlite3_column_blob(stmt, 2),
                      Int(sqlite3_column_bytes(stmt, 2)) == n * Self.embeddingDimension * 4 else { continue }
                let flat = [Float32](UnsafeBufferPointer(
                    start: blob.assumingMemoryBound(to: Float32.self),
                    count: n * Self.embeddingDimension))
                let embs = (0..<n).map { f in
                    Array(flat[f * Self.embeddingDimension ..< (f + 1) * Self.embeddingDimension])
                }
                out.append((id, embs))
            }
            return out
        }
        for (id, embs) in rows { body(id, embs) }
    }

    public var count: Int {
        queue.sync {
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM assets;", -1, &stmt, nil) == SQLITE_OK,
                  sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
            return Int(sqlite3_column_int(stmt, 0))
        }
    }

    // MARK: - helpers

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    private static func epoch(_ date: Date?) -> Double {
        date?.timeIntervalSince1970 ?? 0
    }

    private func exec(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        guard sqlite3_exec(db, sql, nil, nil, &err) == SQLITE_OK else {
            let msg = err.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(err)
            throw ScanCacheError.sqlFailed(msg)
        }
    }
}
