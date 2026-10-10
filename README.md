# Mini-DPVO
A miniature version of [DPVO](https://github.com/princeton-vl/DPVO). With a Gradio web UI (embedded Rerun viewer), ported to AMD GPUs via ROCm/HIP
<p align="center">
  <img src="media/mini-dpvo.gif" alt="example output" width="720" />
</p>

## Installation
This fork runs on AMD GPUs through ROCm/HIP. The native kernels are compiled at install time against the ROCm PyTorch already in your environment, so install the prerequisites first:

1. A ROCm build of PyTorch, e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14`
2. A ROCm build of [lietorch](https://github.com/gandres42/lietorch), compiled against that same torch (`pip install --no-build-isolation .` from a recursive clone).

Then install mini-dpvo from a checkout or straight from git:
```bash
pip install .
```
Extra: `pip install ".[eval]"` for evo-based plotting.

The GPU arch is detected from the first visible GPU. GPUs without official ROCm support need the same override at build time and at runtime, e.g. for an RX 6700 XT:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```
With no GPU visible (e.g. inside `docker build`), set `PYTORCH_ROCM_ARCH=gfx1030` (or your target) instead.

The DPVO and DUSt3R weights are downloaded from the Hugging Face Hub on first use.  Optionally fetch the example videos, which the app lists as examples when run from the directory containing `data/`:
```bash
hf download pablovela5620/dpvo-example-data movies.zip --repo-type dataset --local-dir . && python -m zipfile -e movies.zip . && rm movies.zip
hf download pablovela5620/dpvo-example-data pool.MOV --repo-type dataset --local-dir data/iphone/
```

## Demo
```bash
python tools/gradio_app.py
```

## Acknowledgements
Original Code from [mini-dpvo](https://github.com/pablovela5620/mini-dpvo)

Original Code and paper from [DPVO](https://github.com/princeton-vl/DPVO)
```
@article{teed2023deep,
  title={Deep Patch Visual Odometry},
  author={Teed, Zachary and Lipson, Lahav and Deng, Jia},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
