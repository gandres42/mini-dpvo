# Mini-DPVO
A miniature version of [DPVO](https://github.com/princeton-vl/DPVO). With a Gradio web UI (embedded Rerun viewer), ported to AMD GPUs via ROCm/HIP
<p align="center">
  <img src="media/mini-dpvo.gif" alt="example output" width="720" />
</p>


## Installation
This fork runs on AMD GPUs through ROCm/HIP. The native kernels are compiled at
install time against the ROCm PyTorch already in your environment, so install
the prerequisites first:

1. A ROCm build of PyTorch, e.g.
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14`
2. A ROCm build of [lietorch](https://github.com/gandres42/lietorch), compiled against
   that same torch (`pip install --no-build-isolation .` from a recursive clone). Do not
   use the `lietorch` on PyPI; it is an unrelated CUDA build.

Then install mini-dpvo from a checkout or straight from git:
```bash
pip install .
pip install "git+<repo-url>"
```
Extra: `pip install ".[eval]"` for evo-based plotting.

The GPU arch is detected from the first visible GPU. GPUs without official ROCm
support need the same override at build time and at runtime, e.g. for an RX 6700 XT:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```
With no GPU visible (e.g. inside `docker build`), set `PYTORCH_ROCM_ARCH=gfx1030`
(or your target) instead.

The DPVO and DUSt3R weights are downloaded from the Hugging Face Hub on first use.
Optionally fetch the example videos, which the app lists as examples when run from
the directory containing `data/`:
```bash
hf download pablovela5620/dpvo-example-data movies.zip --repo-type dataset --local-dir . && python -m zipfile -e movies.zip . && rm movies.zip
hf download pablovela5620/dpvo-example-data pool.MOV --repo-type dataset --local-dir data/iphone/
```

## Notes
- The HIP sources are `mini_dpvo/altcorr/correlation_kernel.hip` and `mini_dpvo/fastba/ba_cuda.hip`. The extension modules keep their original names (`mini_dpvo._cuda_corr`, `mini_dpvo._cuda_ba`) because PyTorch exposes ROCm devices through the `torch.cuda` API.
- `torch` is deliberately not a build requirement. pip's isolated build would fetch the CUDA wheel from PyPI, so the in-tree backend in `build_backend/` lets the build see the torch you already have installed.
- `torch_scatter` was replaced by native PyTorch helpers in `mini_dpvo/scatter_utils.py` to avoid ABI coupling to a specific torch build.

## Demo
Hosted Demos can be found in either lightning studio or huggingface spaces

<a target="_blank" href="https://lightning.ai/pablovelagomez1/studios/mini-dpvo">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

<a href='https://huggingface.co/spaces/pablovela5620/MiniDPVO'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>


To run the Gradio app locally (then open http://127.0.0.1:7860)
```
mini-dpvo-app
```
or, from a checkout, `python tools/gradio_app.py`.


## Acknowledgements
Original Code and paper from [DPVO](https://github.com/princeton-vl/DPVO)
```
@article{teed2023deep,
  title={Deep Patch Visual Odometry},
  author={Teed, Zachary and Lipson, Lahav and Deng, Jia},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
