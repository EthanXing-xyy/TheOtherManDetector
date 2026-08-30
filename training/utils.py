"""Shared deterministic training and export utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    project_root = path.parent
    for key in ("data_root", "metadata", "cache_dir", "output_dir"):
        value = Path(config["paths"][key])
        if not value.is_absolute():
            config["paths"][key] = str((project_root / value).resolve())
    return config, project_root


def ensure_output_directories(config: dict[str, Any]) -> None:
    for key in ("cache_dir", "output_dir"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def configure_cpu_threads(config: dict[str, Any]) -> int:
    requested = int(config["training"].get("cpu_threads", 0))
    if requested > 0:
        torch.set_num_threads(requested)
    return int(torch.get_num_threads())


def save_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_raise(name: str, array: np.ndarray | torch.Tensor) -> None:
    finite = torch.isfinite(array).all().item() if isinstance(array, torch.Tensor) else np.isfinite(array).all()
    if not finite:
        raise ValueError(f"{name} contains NaN or Inf")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
