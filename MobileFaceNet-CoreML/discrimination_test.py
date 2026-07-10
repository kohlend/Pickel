"""
Identity-discrimination test (Step 5, second part) using REAL detection +
5-point alignment, exactly like the app pipeline. Uses InsightFace `buffalo_s`
(SCRFD det_500m + the w600k_mbf recognition head this repo converts) so faces
are aligned the same way the model was trained.

Prereqs:
  pip install insightface onnxruntime opencv-python-headless
  # place det_500m.onnx and w600k_mbf.onnx in ~/.insightface/models/buffalo_s/

Usage:
  python discrimination_test.py img_a1.jpg img_a2.jpg img_b1.jpg ...
  # prints the pairwise cosine-similarity matrix of the largest face in each.

Recorded run (obama×2 = same person, biden = different):
             obama1    obama2     biden
  obama1      1.000     0.719    -0.018
  obama2      0.719     1.000     0.074
   biden     -0.018     0.074     1.000
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main(paths):
    import cv2
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_s",
        root=os.path.expanduser("~/.insightface"),
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    embs, names = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"skip (unreadable): {p}"); continue
        faces = app.get(img)
        if not faces:
            print(f"skip (no face): {p}"); continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embs.append(f.normed_embedding)     # already L2-normalized, 512-d
        names.append(os.path.basename(p))

    print("\nPairwise cosine similarity (aligned faces):")
    print("           " + "  ".join(f"{n[:8]:>8}" for n in names))
    for i, a in enumerate(names):
        row = "  ".join(f"{float(np.dot(embs[i], embs[j])):>8.3f}" for j in range(len(names)))
        print(f"{a[:8]:>8}   {row}")
    print("\nArcFace decision threshold ~0.3-0.4: same person well above, "
          "different people near 0.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python discrimination_test.py face1 face2 [face3 ...]")
    main(sys.argv[1:])
