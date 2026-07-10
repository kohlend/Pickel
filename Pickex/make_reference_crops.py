"""
Reference generator for FacePreprocessor.swift.

Produces the 112x112 aligned face crops the Swift pipeline is supposed to output,
using the *same* InsightFace 5-point similarity alignment the model was trained
on. These PNGs are (a) a visual check that the crop looks sensible and (b) the
ground truth the on-device Swift output should match. Also feeds each crop
through the recognition model to confirm a plausible 512-d embedding
(no NaNs, not all zeros).

Run from repo root, in the MobileFaceNet-CoreML venv:
  python Pickex/make_reference_crops.py <img1> <img2> ...
Requires: insightface, onnxruntime, opencv, with ~/.insightface/models/buffalo_s
(det_500m.onnx + w600k_mbf.onnx) present.
"""
import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "reference_crops")


def main(paths):
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    os.makedirs(OUT_DIR, exist_ok=True)
    app = FaceAnalysis(name="buffalo_s", root=os.path.expanduser("~/.insightface"),
                       providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    print(f"{'image':<16}{'faces':>6}{'crop':>10}{'emb_norm':>10}{'nan':>5}{'zeros%':>8}")
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{os.path.basename(p):<16}  unreadable"); continue
        faces = app.get(img)
        if not faces:
            print(f"{os.path.basename(p):<16}{0:>6}   no face"); continue
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))

        # 5-point similarity alignment to the ArcFace 112 template — identical to
        # what FacePreprocessor.swift computes on device.
        aligned = face_align.norm_crop(img, landmark=f.kps, image_size=112)  # BGR, 112x112
        name = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(OUT_DIR, f"{name}_aligned112.png")
        cv2.imwrite(out, aligned)

        e = f.normed_embedding
        nan = bool(np.isnan(e).any())
        zeros = float(np.mean(np.abs(e) < 1e-8)) * 100
        print(f"{name:<16}{len(faces):>6}{'112x112':>10}{np.linalg.norm(f.embedding):>10.2f}"
              f"{str(nan):>5}{zeros:>7.1f}%")

    print(f"\nwrote aligned crops to {OUT_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python make_reference_crops.py img1 [img2 ...]")
    main(sys.argv[1:])
