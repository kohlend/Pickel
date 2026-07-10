"""
Emit a reference MobileFaceNet ONNX (random-but-fixed weights) with the EXACT
production I/O contract: input "data" [1,3,112,112] float, output "fc1" [1,512]
float. This lets convert.py / verify.py be proven end-to-end without the
license-gated production weights. Replace with real w600k_mbf.onnx for shipping.

Usage:  python make_reference_onnx.py --out models/reference_mbf.onnx
"""
import argparse
import torch
from mobilefacenet import MobileFaceNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/reference_mbf.onnx")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = MobileFaceNet().eval()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["data"], output_names=["fc1"],
        opset_version=13, dynamo=False,
        # keep BN as explicit nodes (don't fold into conv bias constants) so the
        # ONNX->torch step stays robust; mirrors how insightface exports w600k_mbf.
        do_constant_folding=False,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {args.out}  ({n/1e6:.2f}M params, IO: data[1,3,112,112] -> fc1[1,512])")


if __name__ == "__main__":
    main()
