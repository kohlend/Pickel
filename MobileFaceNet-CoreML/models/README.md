# models/

`FaceEmbedding_fp16.mlpackage` (committed) is built from the **reference**
MobileFaceNet (deterministic random weights, seed 0) — a runnable example so you
can open it in Xcode and confirm the I/O wiring (`face_image` image input →
`embedding` [1,512]). It is **not** a trained model; its embeddings are
meaningless.

For the production model, fetch weights and rebuild (see ../README.md):

```bash
python download_model.py                       # -> w600k_mbf.onnx
python convert.py --onnx models/w600k_mbf.onnx --precision both
```

`*.onnx`, `*.zip`, and the fp32 reference package are gitignored (reproducible).
