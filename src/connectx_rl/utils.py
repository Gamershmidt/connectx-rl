from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def set_torch_threads(n: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:  # pragma: no cover
        pass


def env_info() -> dict:
    import torch

    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(), processor=platform.processor())


def dump_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1))
