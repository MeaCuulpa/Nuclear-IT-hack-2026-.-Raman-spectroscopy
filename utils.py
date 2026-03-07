from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(data: Any, filepath: str | Path) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class ConfigNode(SimpleNamespace):
    @classmethod
    def from_dict(cls, data: Any):
        if isinstance(data, dict):
            return cls(**{key: cls.from_dict(value) for key, value in data.items()})
        if isinstance(data, list):
            return [cls.from_dict(value) for value in data]
        return data


def load_config(filepath: str | Path):
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ConfigNode.from_dict(data)
