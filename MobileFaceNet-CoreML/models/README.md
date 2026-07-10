# models/

`FaceEmbedding_fp16.mlpackage` (committed) is the **production** model — built
from InsightFace `w600k_mbf.onnx` (WebFace600K, **non-commercial**; see
../README.md §1). Float16 weights, 6.90 MB. Input `face_image` (RGB 112×112) →
`embedding` [1,512], preprocessing baked in. Drag it into Xcode as-is.

Rebuild / produce the fp32 variant:

```bash
python download_model.py                        # -> w600k_mbf.onnx (sha256 9cc6e4a7…)
python convert.py --onnx models/w600k_mbf.onnx --precision both
```

`*.onnx`, `*.zip`, and the fp32 package are gitignored (reproducible, larger).
An offline reference model (random weights) can be regenerated with
`make_reference_onnx.py` for download-free testing.
