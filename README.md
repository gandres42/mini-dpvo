# Mini-DPVO
A miniature version of [DPVO](https://github.com/princeton-vl/DPVO). Using Rerun viewer, Pixi and Gradio for easy use
<p align="center">
  <img src="media/mini-dpvo.gif" alt="example output" width="720" />
</p>


## Installation
Easily installable via [Pixi](https://pixi.sh/latest/).
```bash
git clone https://github.com/pablovela5620/mini-dpvo.git
cd mini-dpvo
pixi install
pixi run post-install
```

## Recent Update Notes

- `mini-dpvo` now uses the prebuilt `lietorch` package from Pixi instead of building the vendored copy in-tree.
- `lietorch` is ABI-coupled to the exact PyTorch/libtorch build. If `pixi` upgrades `pytorch` without a matching `lietorch` build, imports will fail with undefined-symbol errors from `lietorch_backends`.
- The current working combination is driven by `pyproject.toml`/`pixi.lock`; avoid changing `pytorch`, `libtorch`, `torchvision`, or `lietorch` independently.
- The package still builds its own CUDA extensions (`mini_dpvo._cuda_corr` and `mini_dpvo._cuda_ba`) locally through `pixi run post-install`.
- Example assets and checkpoints are now fetched through `hf download`, not `huggingface-cli`.
- `pixi run demo` has been validated on an RTX 5090 with the current lockfile.
- `pixi run rerun` and GUI flows require a valid display session. In headless shells, Rerun will fail unless `DISPLAY`/`XAUTHORITY` are set.

## Demo
Hosted Demos can be found in either lightning studio or huggingface spaces

<a target="_blank" href="https://lightning.ai/pablovelagomez1/studios/mini-dpvo">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

<a href='https://huggingface.co/spaces/pablovela5620/MiniDPVO'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>


To run the gradio frontend
```
pixi run app
```

To run using just the rerun visualizer
```
pixi run rerun
```

To run the full demo pipeline
```
pixi run demo
```

look in the `pixi.toml` file to see exactly what each command does under `tasks`


## Acknowledgements
Original Code and paper from [DPVO](https://github.com/princeton-vl/DPVO)
```
@article{teed2023deep,
  title={Deep Patch Visual Odometry},
  author={Teed, Zachary and Lipson, Lahav and Deng, Jia},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
