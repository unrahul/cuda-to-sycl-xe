"""Build one small extension; the caller configures compiler and SDK runtime."""
from pathlib import Path

from torch.utils.cpp_extension import load


def build(backend="affine", include=None, ldflags=()):
    if backend not in {"affine", "mkl", "dnnl"}:
        raise ValueError("backend must be affine, mkl or dnnl")
    if backend != "affine" and (include is None or not ldflags):
        raise ValueError("library examples require explicit SDK include and link arguments")
    return load(
        name=f"xe_example_{backend}",
        sources=[str(Path(__file__).with_name(f"{backend}.sycl"))],
        with_sycl=True,
        extra_sycl_cflags=["-O2"],
        extra_include_paths=[] if include is None else [include],
        extra_ldflags=list(ldflags),
        verbose=True,
    )
