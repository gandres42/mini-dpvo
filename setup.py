"""Build script for the mini_dpvo native HIP (ROCm) extensions.

Package metadata lives in pyproject.toml. The kernels are compiled against the
ROCm build of PyTorch installed in the target environment, so install torch
first, then:

    pip install .

The GPU architecture is taken from PYTORCH_ROCM_ARCH (e.g. "gfx1030") or, if
unset, from the first visible GPU. That honours HSA_OVERRIDE_GFX_VERSION, so
build with the same override you run with. Set PYTORCH_ROCM_ARCH when building
without a GPU (e.g. during `docker build`).
"""

import os

from setuptools import setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as e:
    raise RuntimeError(
        "mini-dpvo compiles its kernels against the installed PyTorch; install "
        "a ROCm build of torch first (https://pytorch.org/get-started/locally/)"
    ) from e

if torch.version.hip is None:
    raise RuntimeError(
        f"mini-dpvo requires a ROCm build of PyTorch, found torch {torch.__version__}"
    )

if not os.environ.get("PYTORCH_ROCM_ARCH"):
    # Without a visible GPU torch would pass an empty --offload-arch and the
    # kernels would fail at runtime with "device kernel image is invalid".
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm GPU visible to detect the target arch; set PYTORCH_ROCM_ARCH "
            "(e.g. PYTORCH_ROCM_ARCH=gfx1030)"
        )
    # Reflects HSA_OVERRIDE_GFX_VERSION, so the kernels match what the runtime loads.
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    os.environ["PYTORCH_ROCM_ARCH"] = arch
    print(f"Auto-detected ROCm arch: {arch}")

setup(
    ext_modules=[
        CUDAExtension(
            "mini_dpvo._cuda_corr",
            sources=[
                "mini_dpvo/altcorr/correlation.cpp",
                "mini_dpvo/altcorr/correlation_kernel.hip",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CUDAExtension(
            "mini_dpvo._cuda_ba",
            sources=["mini_dpvo/fastba/ba.cpp", "mini_dpvo/fastba/ba_cuda.hip"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
