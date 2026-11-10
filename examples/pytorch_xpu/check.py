"""Small correctness smoke, with host FP64 arithmetic outside the extension."""
import argparse
import json
import math
import os

import torch

from loader import build


def reference(a, b):
    if b is None:
        return a.double() * 2 + 1
    rows, cols = a.double().tolist(), b.double().t().tolist()
    return torch.tensor(
        [[math.fsum(x * y for x, y in zip(row, col)) for col in cols]
         for row in rows], dtype=torch.float64,
    )


def check(got, expected):
    actual = got.cpu().double()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    return (actual - expected).abs().max().item() if actual.numel() else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("affine", "mkl", "dnnl"))
    parser.add_argument("--device", required=True, help="caller-selected XPU device")
    args = parser.parse_args()
    if not torch.xpu.is_available():
        raise RuntimeError("XPU runtime is unavailable")
    device = torch.device(args.device)
    if device.type != "xpu":
        raise ValueError("select an XPU device explicitly")
    torch.xpu.set_device(device)
    options = {}
    if args.backend != "affine":
        prefix = f"XPU_{args.backend.upper()}"
        options = {"include": os.environ[f"{prefix}_INCLUDE"],
                   "ldflags": json.loads(os.environ[f"{prefix}_LDFLAGS"])}
    extension = build(args.backend, **options)
    generator = torch.Generator().manual_seed(27)
    shapes = [(0,), (1,), (37,)] if args.backend == "affine" else [(1, 1, 1), (7, 13, 5)]
    stream = torch.xpu.Stream(device=device)
    max_error = 0.0
    for shape in shapes:
        a = torch.randn(shape if args.backend == "affine" else shape[:2],
                        generator=generator, dtype=torch.float32) * 0.2
        b = None if args.backend == "affine" else torch.randn(
            (shape[1], shape[2]), generator=generator, dtype=torch.float32) * 0.2
        expected = reference(a, b)
        # Producers, extension calls and consumers all run on a non-default stream.
        with torch.xpu.stream(stream):
            # Real nn.Parameters (requires_grad=True), passed as a model's forward would pass them.
            x = torch.nn.Parameter(a.to(device).clone())
            inputs = [x] if b is None else [x, torch.nn.Parameter(b.to(device).clone())]
            first = extension.run(*inputs)
            second = extension.run(*inputs)
            consumed = second * 3.0 - 2.0
            temporary = extension.run(*[t.detach().clone() for t in inputs])
        stream.synchronize()
        for result in (first, second, temporary):
            max_error = max(max_error, check(result, expected))
        check(consumed, expected * 3 - 2)
        for original, candidate in zip([a] if b is None else [a, b], inputs):
            assert torch.equal(original, candidate.detach().cpu()), "input changed"
    print(f"PASS {args.backend}: nondefault stream, FP64 reference, retained outputs, "
          f"temporary inputs, unchanged inputs; max_abs_error={max_error:.9g}")


if __name__ == "__main__":
    main()
