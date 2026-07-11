"""
Verification harness for LibraryScanner.swift (Step C).

Mirrors the scanner's matching logic — embed ALL faces per photo (largest
first), photo matches if ANY face's cosine vs the reference profile >= 0.40 —
and runs it over a real test "library" (LFW photos): 8 positives (Bush, none
of them reference photos), 10 single-face negatives, 1 multi-face group-photo
negative. Prints per-photo best-face scores and precision/recall.

Run in the MobileFaceNet-CoreML venv (buffalo_s in ~/.insightface).
Library photos in /tmp/library, reference photos /tmp/rf/bush{1..4}.jpg
(see README for how the sets were fetched).
"""
import glob
import os
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

MATCH_THRESHOLD = 0.40      # == LibraryScanner.defaultMatchThreshold
MAX_FACES_PER_PHOTO = 8     # == LibraryScannerConfig.maxFacesPerPhoto


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


def all_face_embeddings(app, path, max_faces=MAX_FACES_PER_PHOTO):
    img = cv2.imread(path)
    if img is None:
        return []
    faces = app.get(img)
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    return [f.embedding.astype(np.float64) for f in faces[:max_faces]]


def main():
    app = build_app()

    # Reference profile (Step B): mean of normalized -> normalize.
    refs = [all_face_embeddings(app, f"/tmp/rf/bush{i}.jpg")[0] for i in (1, 2, 3, 4)]
    profile = l2(np.mean([l2(r) for r in refs], axis=0))

    library = sorted(glob.glob("/tmp/library/*.jpg")) + ["/tmp/faces/t1.jpg"]  # + group photo
    tp = fp = fn = tn = 0
    print(f"{'photo':<22}{'faces':>6}{'best':>8}  match?  expected")
    for p in library:
        name = os.path.basename(p)
        embs = all_face_embeddings(app, p)
        best = max((float(np.dot(profile, l2(e))) for e in embs), default=None)
        is_match = best is not None and best >= MATCH_THRESHOLD
        expected = name.startswith("pos_")
        ok = "✓" if is_match == expected else "✗ MISMATCH"
        b = f"{best:.3f}" if best is not None else "—"
        print(f"{name:<22}{len(embs):>6}{b:>8}  {'YES' if is_match else 'no ':<5} {'pos' if expected else 'neg'}  {ok}")
        if expected and is_match: tp += 1
        elif expected: fn += 1
        elif is_match: fp += 1
        else: tn += 1

    print(f"\nthreshold {MATCH_THRESHOLD}: recall {tp}/{tp+fn}, "
          f"false positives {fp}/{fp+tn}  "
          f"({'ALL CORRECT' if fp == 0 and fn == 0 else 'errors above'})")


if __name__ == "__main__":
    main()
