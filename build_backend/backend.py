"""PEP 517 backend: setuptools, built against the PyTorch already installed.

The HIP kernels must be compiled against the exact ROCm build of torch that
will load them. That build comes from download.pytorch.org, not PyPI, so it
cannot be listed in ``build-system.requires`` (pip would fetch the CUDA wheel
instead). pip's build isolation hides the installed torch from the build, so
this backend appends the target interpreter's own ``sys.path`` after the
isolated entries before handing off to setuptools. Build tools (setuptools,
ninja) still come from the isolated environment.
"""

import json
import os
import subprocess
import sys


def _expose_installed_torch():
    try:
        import torch  # noqa: F401  (already visible, e.g. --no-build-isolation)

        return
    except ImportError:
        pass

    # pip isolates via PYTHONPATH + PYTHONNOUSERSITE; without them the same
    # interpreter reports the sys.path of the environment being installed into.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTHONNOUSERSITE")
    }
    result = subprocess.run(
        [sys.executable, "-c", "import json, sys; print(json.dumps(sys.path[1:]))"],
        env=env,
        cwd=os.path.expanduser("~"),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return
    for path in json.loads(result.stdout):
        if path and path not in sys.path:
            sys.path.append(path)


_expose_installed_torch()

from setuptools.build_meta import *  # noqa: E402,F401,F403
from setuptools.build_meta import __all__  # noqa: E402,F401
