import argparse
import numpy as np
import torch
import onnxruntime as ort

from core.models import build_model, load_model_weights


# Verify ONNX file with original checkpoints file.
def verify(ckpt, version, model_size, onnx_file, width, height, max_disp, input_layout="nchw"):
    # load models
    ## Original model
    torch.manual_seed(0)
    left = torch.rand(1, 3, height, width, dtype=torch.float32)
    right = torch.rand(1, 3, height, width, dtype=torch.float32)

    model = build_model(version, model_size=model_size, max_disp=max_disp)
    load_model_weights(model, torch.load(ckpt, map_location="cpu"), strict=True)
    model.eval()

    ## ONNX model
    sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

    # run models
    ## Original model
    with torch.no_grad():
        out_torch = model(left, right, max_disp=max_disp, test_mode=True)

    ## ONNX model
    if input_layout == "nhwc":
        left_onnx = left.permute(0, 2, 3, 1).contiguous().numpy()
        right_onnx = right.permute(0, 2, 3, 1).contiguous().numpy()
    else:
        left_onnx = left.numpy()
        right_onnx = right.numpy()
    out_onnx = sess.run(
        ["disparity"], {"left": left_onnx, "right": right_onnx}
    )[0]

    # compare
    a = out_torch.numpy().astype(np.float64)
    b = out_onnx.astype(np.float64)
    diff = np.abs(a - b)
    print("torch disparity:", a.min(), a.max())
    print("onnx  disparity:", b.min(), b.max())
    print("max abs diff   :", diff.max())
    print("mean abs diff  :", diff.mean())
    print("allclose(1e-4):", np.allclose(a, b, atol=1e-4, rtol=1e-3))


def parse_args():
    parser = argparse.ArgumentParser(description="Verify ONNX export against original PyTorch model.")
    parser.add_argument("--ckpt", default="./checkpoints/LAS2_M.pth", help="path to .pth checkpoint file")
    parser.add_argument("--version", default="las2", help="model version, e.g. las1 or las2")
    parser.add_argument("--model_size", default="m", help="LAS2 model size: s, m, l, or h")
    parser.add_argument("--onnx_file", default="liteanystereo.onnx", help="path to exported ONNX file")
    parser.add_argument("--width", type=int, default=1248, help="input width, per single image (not left+right combined)")
    parser.add_argument("--height", type=int, default=384, help="input height")
    parser.add_argument("--max_disp", type=int, default=192, help="maximum disparity used by the model")
    parser.add_argument(
        "--input_layout",
        choices=["nchw", "nhwc"],
        default="nchw",
        help="layout used by the ONNX input tensors",
    )
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    verify(ckpt=args.ckpt,
           version=args.version,
           model_size=args.model_size,
           onnx_file=args.onnx_file,
           width=args.width,
           height=args.height,
           max_disp=args.max_disp,
           input_layout=args.input_layout)
