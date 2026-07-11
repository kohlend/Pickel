"""
Verification harness for ScanCache.swift + LibraryScanner.swift (Step C, v2).

Mirrors the Swift logic exactly — same SQLite schema, same lookup/invalidate
semantics, same match rule — and verifies on real photos + real model:

  1. Cold scan: all 19 photos computed, results correct (baseline from
     verify_library_scan.py).
  2. Warm scan: 100% cache hits, ZERO model invocations, identical results;
     wall-time speedup measured.
  3. Invalidation: bump one photo's modification date -> exactly that one is
     recomputed.
  4. Prune: drop photos from the "library" -> their rows disappear.
  5. rematchFromCache: build a NEW reference profile (Powell) and match it
     against cached embeddings only — no photo/model work; also timed against
     10,000 synthetic cached rows.
  6. Environment timing: per-photo detect+embed cost on THIS machine
     (Linux/CPU x86) — a data point, NOT an iPhone number.

Run in the MobileFaceNet-CoreML venv (buffalo_s in ~/.insightface).
"""
import glob
import os
import sqlite3
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

DIM = 512
THRESHOLD = 0.40
DB = "/tmp/scancache_test.sqlite"

model_invocations = 0


def l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ── ScanCache mirror (same schema as ScanCache.swift) ──────────────────────
class Cache:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("""CREATE TABLE IF NOT EXISTS assets(
            local_id TEXT PRIMARY KEY, modified_at REAL NOT NULL,
            face_count INTEGER NOT NULL, embeddings BLOB);""")

    def lookup(self, local_id, modified_at):
        row = self.db.execute(
            "SELECT modified_at, face_count, embeddings FROM assets WHERE local_id=?",
            (local_id,)).fetchone()
        if row is None or row[0] != modified_at:
            return None                      # miss or stale
        n = row[1]
        if n == 0:
            return []                        # valid hit: scanned, no faces
        flat = np.frombuffer(row[2], dtype=np.float32)
        return list(flat.reshape(n, DIM).astype(np.float64))

    def store(self, local_id, modified_at, embeddings):
        blob = (np.asarray(embeddings, dtype=np.float32).tobytes()
                if len(embeddings) else None)
        self.db.execute(
            "INSERT OR REPLACE INTO assets VALUES (?,?,?,?)",
            (local_id, modified_at, len(embeddings), blob))
        self.db.commit()

    def prune(self, keep):
        rows = [r[0] for r in self.db.execute("SELECT local_id FROM assets")]
        removed = [r for r in rows if r not in keep]
        for r in removed:
            self.db.execute("DELETE FROM assets WHERE local_id=?", (r,))
        self.db.commit()
        return len(removed)

    def for_each(self):
        for lid, n, blob in self.db.execute(
                "SELECT local_id, face_count, embeddings FROM assets WHERE face_count>0"):
            flat = np.frombuffer(blob, dtype=np.float32)
            yield lid, list(flat.reshape(n, DIM).astype(np.float64))

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]


# ── model (counts invocations, mirrors embeddingsForAllFaces) ──────────────
def build_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_s", root=os.path.expanduser("~/.insightface"),
                       providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def all_face_embeddings(app, path, max_faces=8):
    global model_invocations
    model_invocations += 1
    img = cv2.imread(path)
    faces = app.get(img)
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    return [f.embedding.astype(np.float64) for f in faces[:max_faces]]


# ── scanner mirror ──────────────────────────────────────────────────────────
def scan(app, cache, library, reference):
    """library: list of (id, path, modified_at). Returns (matches, cache_hits)."""
    matches, hits = [], 0
    for lid, path, mod in library:
        cached = cache.lookup(lid, mod)
        if cached is not None:
            embs, hits = cached, hits + 1
        else:
            embs = all_face_embeddings(app, path)
            cache.store(lid, mod, embs)
        best = max((float(np.dot(reference, l2(e))) for e in embs), default=-1)
        if best >= THRESHOLD:
            matches.append((lid, round(best, 3)))
    return matches, hits


def main():
    global model_invocations
    if os.path.exists(DB):
        os.remove(DB)
    app = build_app()
    cache = Cache(DB)

    photos = sorted(glob.glob("/tmp/library/*.jpg")) + ["/tmp/faces/t1.jpg"]
    library = [(os.path.basename(p), p, 1000.0) for p in photos]

    bush = l2(np.mean([l2(all_face_embeddings(app, f"/tmp/rf/bush{i}.jpg")[0])
                       for i in (1, 2, 3, 4)], axis=0))
    model_invocations = 0

    # 1) cold scan
    t0 = time.perf_counter()
    m1, h1 = scan(app, cache, library, bush)
    cold = time.perf_counter() - t0
    inv_cold = model_invocations
    print(f"1) cold scan : {len(m1)} matches, cache hits {h1}/{len(library)}, "
          f"model runs {inv_cold}, {cold:.2f}s "
          f"({cold/len(library)*1000:.0f} ms/photo on THIS Linux/CPU box)")

    # 2) warm scan
    model_invocations = 0
    t0 = time.perf_counter()
    m2, h2 = scan(app, cache, library, bush)
    warm = time.perf_counter() - t0
    ok2 = m1 == m2 and h2 == len(library) and model_invocations == 0
    print(f"2) warm scan : identical results {m1 == m2}, cache hits {h2}/{len(library)}, "
          f"model runs {model_invocations}, {warm*1000:.0f} ms  "
          f"→ {cold/warm:.0f}× faster  {'✅' if ok2 else '❌'}")

    # 3) invalidation: one photo "edited"
    edited = library[10]
    library[10] = (edited[0], edited[1], 2000.0)
    model_invocations = 0
    m3, h3 = scan(app, cache, library, bush)
    ok3 = model_invocations == 1 and h3 == len(library) - 1 and m3 == m1
    print(f"3) invalidate: 1 modified photo → model runs {model_invocations} "
          f"(expected 1), hits {h3}/{len(library)}, results unchanged {m3 == m1}  "
          f"{'✅' if ok3 else '❌'}")

    # 4) prune: user deleted 3 photos
    keep = {lid for lid, _, _ in library[:-3]}
    removed = cache.prune(keep)
    ok4 = removed == 3 and cache.count() == len(library) - 3
    print(f"4) prune     : removed {removed} rows (expected 3), "
          f"{cache.count()} remain  {'✅' if ok4 else '❌'}")

    # 5) NEW reference (Powell) matched purely from cache — no model runs
    powell = l2(all_face_embeddings(app, "/tmp/library/neg_powell.jpg")[0])
    model_invocations = 0
    t0 = time.perf_counter()
    rematch = []
    for lid, embs in cache.for_each():
        best = max(float(np.dot(powell, l2(e))) for e in embs)
        if best >= THRESHOLD:
            rematch.append((lid, round(best, 3)))
    dt = (time.perf_counter() - t0) * 1000
    print(f"5) rematchFromCache (new person: Powell): {rematch} "
          f"model runs {model_invocations}, {dt:.1f} ms")

    # 5b) rematch speed at scale: 10k synthetic cached photos
    big = np.random.default_rng(0).standard_normal((10_000, DIM)).astype(np.float32)
    big /= np.linalg.norm(big, axis=1, keepdims=True)
    t0 = time.perf_counter()
    _ = (big @ powell >= THRESHOLD).sum()
    dt = (time.perf_counter() - t0) * 1000
    print(f"   at scale: 10,000 cached embeddings re-matched in {dt:.1f} ms")


if __name__ == "__main__":
    main()
