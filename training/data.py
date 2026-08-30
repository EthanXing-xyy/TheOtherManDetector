"""Metadata-driven dataset validation, preprocessing, and cache access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing.audio import create_slaney_mel_filterbank, load_and_extract
from .preprocessing.imu import block_imu_target, load_imu, normalize_imu


EXPECTED_LABELS = {
    "train": {"NORMAL"},
    "validation": {"NORMAL"},
    "normal_test": {"NORMAL"},
    "slight_anomaly": {"SLIGHT_ANOMALY"},
    "fierce_anomaly": {"FIERCE_ANOMALY"},
}


def load_metadata(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["paths"]["metadata"])
    frame = pd.read_csv(path, dtype={"sample_id": str})
    required = {
        "dataset",
        "sample_id",
        "label_type",
        "status",
        "rejection_reason",
        "imu_output_file",
        "audio_output_file",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    frame["status"] = frame["status"].astype(str).str.strip()
    return frame


def rows_for_split(
    config: dict[str, Any], metadata: pd.DataFrame, split: str
) -> pd.DataFrame:
    split_groups = config["data"]["splits"]
    if split not in split_groups:
        raise KeyError(f"Unknown split: {split}")
    rows = metadata.loc[
        (metadata["status"] == config["data"]["accepted_status"])
        & metadata["dataset"].isin(split_groups[split])
    ].copy()
    return rows.sort_values(["dataset", "sample_id"]).reset_index(drop=True)


def audit_splits(config: dict[str, Any], metadata: pd.DataFrame) -> dict[str, Any]:
    accepted_status = config["data"]["accepted_status"]
    accepted = metadata.loc[metadata["status"] == accepted_status].copy()
    rejected = metadata.loc[metadata["status"] != accepted_status].copy()
    expected_counts = config["data"]["expected_counts"]
    audit: dict[str, Any] = {
        "metadata_rows": int(len(metadata)),
        "accepted_rows": int(len(accepted)),
        "rejected_rows": int(len(rejected)),
        "splits": {},
        "rejected_samples": rejected[
            ["dataset", "sample_id", "status", "rejection_reason"]
        ].fillna("").to_dict(orient="records"),
    }
    seen: set[tuple[str, str]] = set()
    for split in config["data"]["splits"]:
        rows = rows_for_split(config, metadata, split)
        expected = int(expected_counts[split])
        if len(rows) != expected:
            raise ValueError(f"Split {split}: expected {expected}, found {len(rows)}")
        labels = set(rows["label_type"].astype(str))
        if labels != EXPECTED_LABELS[split]:
            raise ValueError(
                f"Split {split}: expected labels {EXPECTED_LABELS[split]}, found {labels}"
            )
        keys = set(zip(rows["dataset"].astype(str), rows["sample_id"].astype(str)))
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(f"Split overlap detected: {sorted(overlap)[:3]}")
        seen.update(keys)
        audit["splits"][split] = {
            "groups": list(config["data"]["splits"][split]),
            "count": int(len(rows)),
            "labels": rows["label_type"].value_counts().to_dict(),
        }
    if len(accepted) != int(expected_counts["total_accepted"]):
        raise ValueError("Accepted sample total does not match the fixed dataset contract")
    if len(rejected) != int(expected_counts["rejected"]):
        raise ValueError("Rejected sample total does not match the cleaning records")
    if len(seen) != len(accepted):
        raise ValueError("Accepted samples are not assigned to exactly one split")
    numeric_contract = {
        "imu_sample_count": 720,
        "audio_sample_count": 115200,
        "imu_sample_rate": 100.0,
        "audio_sample_rate": 16000.0,
        "audio_channels": 1,
        "audio_sample_width_bytes": 2,
    }
    for column, expected in numeric_contract.items():
        values = pd.to_numeric(accepted[column]).to_numpy()
        if not np.allclose(values, expected, atol=1e-9, rtol=0.0):
            raise ValueError(f"Processed metadata violates {column}={expected}")
    return audit


def resolve_sample_paths(
    config: dict[str, Any], row: pd.Series
) -> tuple[Path, Path]:
    root = Path(config["paths"]["data_root"]).resolve()
    imu_relative = Path(str(row["imu_output_file"]))
    audio_relative = Path(str(row["audio_output_file"]))
    if imu_relative.is_absolute() or audio_relative.is_absolute():
        raise ValueError("Metadata sample paths must be relative")
    imu_path = (root / imu_relative).resolve()
    audio_path = (root / audio_relative).resolve()
    if not imu_path.is_relative_to(root) or not audio_path.is_relative_to(root):
        raise ValueError("Metadata sample path escapes the processed dataset root")
    if not imu_path.is_file() or not audio_path.is_file():
        raise FileNotFoundError(f"Missing synchronized sample pair: {imu_path} / {audio_path}")
    return imu_path, audio_path


def compute_train_statistics(
    config: dict[str, Any], train_rows: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Compute population statistics exclusively from Normal Train."""

    if len(train_rows) != 123 or set(train_rows["label_type"]) != {"NORMAL"}:
        raise ValueError("Normalization requires exactly 123 Normal Train samples")
    audio_config = config["audio"]
    filterbank = create_slaney_mel_filterbank(
        int(audio_config["sample_rate"]),
        int(audio_config["n_fft"]),
        int(audio_config["n_mels"]),
        float(audio_config["f_min"]),
        float(audio_config["f_max"]),
    )
    audio_sum = np.zeros(40, dtype=np.float64)
    audio_sumsq = np.zeros(40, dtype=np.float64)
    imu_sum = np.zeros(6, dtype=np.float64)
    imu_sumsq = np.zeros(6, dtype=np.float64)
    audio_count = 0
    imu_count = 0
    for _, row in train_rows.iterrows():
        imu_path, audio_path = resolve_sample_paths(config, row)
        log_mel = load_and_extract(audio_path, audio_config, filterbank)
        imu = load_imu(imu_path, config["imu"])
        audio_sum += log_mel.sum(axis=0, dtype=np.float64)
        audio_sumsq += np.square(log_mel, dtype=np.float64).sum(axis=0)
        imu_sum += imu.sum(axis=0, dtype=np.float64)
        imu_sumsq += np.square(imu, dtype=np.float64).sum(axis=0)
        audio_count += log_mel.shape[0]
        imu_count += imu.shape[0]
    audio_mean = audio_sum / audio_count
    audio_std = np.sqrt(np.maximum(audio_sumsq / audio_count - audio_mean**2, 0.0))
    imu_mean = imu_sum / imu_count
    imu_std = np.sqrt(np.maximum(imu_sumsq / imu_count - imu_mean**2, 0.0))
    if np.any(audio_std <= 0) or np.any(imu_std <= 0):
        raise ValueError("A Normal Train normalization standard deviation is zero")
    provenance = {
        "fit_split": "Normal Train only",
        "fit_datasets": ["normal_train"],
        "fit_sample_count": 123,
        "ddof": 0,
        "per_sample_normalization": False,
    }
    audio_stats = {
        **provenance,
        "feature": "40-band log(mel_energy + 1e-6)",
        "observation_count": int(audio_count),
        "mean": audio_mean.tolist(),
        "std": audio_std.tolist(),
        "mel_scale": "slaney",
        "mel_norm": "slaney",
        "f_min_hz": 0.0,
        "f_max_hz": 7000.0,
        "pcm_scaling": "signed_integer_full_scale",
        "fft_normalization": "none",
        "logarithm": "natural",
    }
    imu_stats = {
        **provenance,
        "feature": "raw six-axis IMU in CSV column order",
        "observation_count": int(imu_count),
        "axis_names": list(config["imu"]["axes"]),
        "mean": imu_mean.tolist(),
        "std": imu_std.tolist(),
    }
    return audio_stats, imu_stats, filterbank


def load_normalization(artifacts_dir: str | Path) -> tuple[np.ndarray, ...]:
    root = Path(artifacts_dir)
    with (root / "audio_normalization.json").open("r", encoding="utf-8") as handle:
        audio = json.load(handle)
    with (root / "imu_normalization.json").open("r", encoding="utf-8") as handle:
        imu = json.load(handle)
    arrays = (
        np.asarray(audio["mean"], dtype=np.float32),
        np.asarray(audio["std"], dtype=np.float32),
        np.asarray(imu["mean"], dtype=np.float32),
        np.asarray(imu["std"], dtype=np.float32),
    )
    if tuple(value.shape for value in arrays) != ((40,), (40,), (6,), (6,)):
        raise ValueError("Normalization artifact shapes are invalid")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Normalization artifacts contain non-finite values")
    if np.any(arrays[1] <= 0) or np.any(arrays[3] <= 0):
        raise ValueError("Normalization artifacts contain non-positive scales")
    return arrays


def cache_path(cache_root: str | Path, dataset: str, sample_id: str) -> Path:
    return Path(cache_root) / dataset / f"{sample_id}.npz"


def preprocess_row(
    config: dict[str, Any],
    row: pd.Series,
    artifacts_dir: str | Path,
) -> dict[str, np.ndarray]:
    audio_mean, audio_std, imu_mean, imu_std = load_normalization(artifacts_dir)
    filterbank = np.load(Path(artifacts_dir) / "mel_filterbank.npy")
    imu_path, audio_path = resolve_sample_paths(config, row)
    log_mel = load_and_extract(audio_path, config["audio"], filterbank)
    raw_imu = load_imu(imu_path, config["imu"])
    audio = ((log_mel - audio_mean[None, :]) / audio_std[None, :]).astype(np.float32)
    imu = normalize_imu(raw_imu, imu_mean, imu_std)
    target = np.concatenate([audio, block_imu_target(imu)], axis=1).astype(np.float32)
    centers = float(config["time_grid"]["first_center_seconds"]) + float(
        config["time_grid"]["block_seconds"]
    ) * np.arange(180, dtype=np.float32)
    if audio.shape != (180, 40) or imu.shape != (720, 6) or target.shape != (180, 64):
        raise AssertionError("Preprocessed sample shapes do not match the model contract")
    return {"audio": audio, "imu": imu, "target": target, "time_centers": centers}


class CachedMultimodalDataset(Dataset):
    """PyTorch dataset backed by compressed, preprocessed sample files."""

    def __init__(self, cache_root: str | Path, rows: pd.DataFrame) -> None:
        self.cache_root = Path(cache_root)
        self.rows = rows.reset_index(drop=True).copy()
        missing = [
            cache_path(self.cache_root, str(row.dataset), str(row.sample_id))
            for row in self.rows.itertuples()
            if not cache_path(self.cache_root, str(row.dataset), str(row.sample_id)).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} cache files; first: {missing[0]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        path = cache_path(self.cache_root, str(row["dataset"]), str(row["sample_id"]))
        with np.load(path) as values:
            audio = np.asarray(values["audio"], dtype=np.float32)
            imu = np.asarray(values["imu"], dtype=np.float32)
            target = np.asarray(values["target"], dtype=np.float32)
        if audio.shape != (180, 40) or imu.shape != (720, 6) or target.shape != (180, 64):
            raise AssertionError(f"Invalid cached sample shapes in {path}")
        return {
            "audio": torch.from_numpy(audio.copy()),
            "imu": torch.from_numpy(imu.copy()),
            "target": torch.from_numpy(target.copy()),
            "dataset": str(row["dataset"]),
            "sample_id": str(row["sample_id"]),
            "label_type": str(row["label_type"]),
        }
