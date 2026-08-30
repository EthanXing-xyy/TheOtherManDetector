"""IMU loading and Normal-Train global per-axis normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["relative_time", "ax", "ay", "az", "gx", "gy", "gz"]


def load_imu(path: str | Path, config: dict[str, Any]) -> np.ndarray:
    if list(config["axes"]) != EXPECTED_COLUMNS[1:]:
        raise ValueError(f"IMU config axis order must remain {EXPECTED_COLUMNS[1:]}")
    frame = pd.read_csv(Path(path), encoding="utf-8-sig")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path}: IMU columns/order changed; expected {EXPECTED_COLUMNS}, got {list(frame.columns)}"
        )
    values = frame[EXPECTED_COLUMNS[1:]].to_numpy(dtype=np.float32)
    expected_shape = (int(config["samples"]), len(config["axes"]))
    if values.shape != expected_shape:
        raise ValueError(f"{path}: expected IMU shape {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite IMU values")
    times = frame["relative_time"].to_numpy(dtype=np.float64)
    expected_times = -2.5 + np.arange(expected_shape[0], dtype=np.float64) / float(config["sample_rate"])
    if not np.allclose(times, expected_times, atol=1e-7, rtol=0.0):
        raise ValueError(f"{path}: IMU physical time grid differs from [-2.5,+4.7) at 100 Hz")
    return values


def normalize_imu(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError(f"Expected six-axis mean/std, got {mean.shape}/{std.shape}")
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("Invalid IMU standard deviations")
    normalized = ((values - mean[None, :]) / std[None, :]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized IMU values")
    return normalized


def block_imu_target(normalized_imu: np.ndarray, frames: int = 180, samples_per_block: int = 4) -> np.ndarray:
    expected = (frames * samples_per_block, 6)
    if normalized_imu.shape != expected:
        raise ValueError(f"Expected normalized IMU shape {expected}, got {normalized_imu.shape}")
    # C-order flattening preserves [t0 axes 0..5, t1 axes 0..5, t2, t3].
    target = normalized_imu.reshape(frames, samples_per_block, 6).reshape(frames, 24)
    if target.shape != (frames, 24):
        raise AssertionError(f"IMU target shape mismatch: {target.shape}")
    return target.astype(np.float32, copy=False)
