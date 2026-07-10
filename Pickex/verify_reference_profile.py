"""
Verification harness for ReferenceProfileBuilder.swift.

Mirrors the Swift logic in Python (same model, same aggregation = normalize-each
→ mean → normalize, same consistency threshold 0.4) and runs the three Step-6
scenarios on real photos, printing the actual similarity numbers.

Run in the MobileFaceNet-CoreML venv (buffalo_s in ~/.insightface).
"""
import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

CONSISTENCY_THRESHOLD = 0.4   # == ReferenceProfileBuilder.defaultConsistencyThreshold


def l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_s", root=os.path.expanduser("~/.insightface"),
                       providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def embed(app, path):
    """Return (raw_vector, face_count) or (None, 0) if no face — mirrors FaceEmbedder."""
    img = cv2.imread(path)
    if img is None:
        return None, 0
    faces = app.get(img)
    if not faces:
        return None, 0
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    return f.embedding.astype(np.float64), len(faces)


def build_profile(app, paths):
    """Mirror of ReferenceProfileBuilder.build(from:)."""
    used, skipped, multi = [], [], []
    for i, p in enumerate(paths):
        vec, n = embed(app, p)
        if vec is None:
            skipped.append((i, "noFaceDetected"))
        else:
            used.append((i, vec))
            if n > 1:
                multi.append(i)

    if not used:
        return {"error": "noFaceInAnyPhoto"}

    # consistency (soft warning)
    normed = [l2(v) for _, v in used]
    flagged, min_sim = [], 1.0
    for a in range(len(normed)):
        for b in range(a + 1, len(normed)):
            s = float(np.dot(normed[a], normed[b]))
            min_sim = min(min_sim, s)
            if s < CONSISTENCY_THRESHOLD:
                flagged.append((used[a][0], used[b][0], s))

    # aggregate: normalize-each -> mean -> normalize
    final = l2(np.mean([l2(v) for _, v in used], axis=0))

    return {
        "usedPhotoCount": len(used),
        "skipped": skipped,
        "multipleFaceWarnings": multi,
        "minSimilarity": min_sim if len(used) >= 2 else None,
        "flaggedPairs": flagged,
        "consistencyWarning": bool(flagged),
        "embeddingNorm": float(np.linalg.norm(final)),
        "embeddingDim": int(final.shape[0]),
        "pairwise": [(used[a][0], used[b][0], float(np.dot(normed[a], normed[b])))
                     for a in range(len(normed)) for b in range(a + 1, len(normed))],
    }


def show(title, paths, res):
    print(f"\n=== {title} ===")
    print("photos:", [os.path.basename(p) for p in paths])
    if "error" in res:
        print("  -> ERROR:", res["error"]); return
    print(f"  used={res['usedPhotoCount']}  skipped={res['skipped']}  "
          f"multipleFaces={res['multipleFaceWarnings']}")
    if res["pairwise"]:
        print("  pairwise cosine:")
        for a, b, s in res["pairwise"]:
            print(f"    photo{a}–photo{b}: {s:+.3f}")
        print(f"  min similarity = {res['minSimilarity']:.3f}  (threshold {CONSISTENCY_THRESHOLD})")
    print(f"  consistencyWarning = {res['consistencyWarning']}"
          + (f"  flagged={[(a,b,round(s,3)) for a,b,s in res['flaggedPairs']]}"
             if res['flaggedPairs'] else ""))
    print(f"  final embedding: dim={res['embeddingDim']}, L2-norm={res['embeddingNorm']:.3f}")


def main():
    app = build_app()

    # a guaranteed no-face image (gradient "landscape")
    land = "/tmp/rf/landscape.png"
    g = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    cv2.imwrite(land, cv2.merge([g, g.T, np.full_like(g, 120)]))

    R = "/tmp/rf"
    s1 = [f"{R}/bush1.jpg", f"{R}/bush2.jpg", f"{R}/bush3.jpg", f"{R}/bush4.jpg"]
    s2 = [f"{R}/bush1.jpg", f"{R}/bush2.jpg", f"{R}/bush3.jpg", f"{R}/powell.jpg"]
    s3 = [f"{R}/bush1.jpg", f"{R}/bush2.jpg", f"{R}/bush3.jpg", land]

    show("Scenario 1 — 4 photos, same person (Bush)", s1, build_profile(app, s1))
    show("Scenario 2 — 1 of 4 is a different person (photo3 = Powell)", s2, build_profile(app, s2))
    show("Scenario 3 — 1 of 4 has no face (photo3 = landscape)", s3, build_profile(app, s3))


if __name__ == "__main__":
    main()
