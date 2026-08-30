#!/usr/bin/env python
"""Create normalized Audio/IMU caches on the fixed deployment input grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.data import (
    audit_splits,
    cache_path,
    compute_train_statistics,
    load_metadata,
    preprocess_row,
    rows_for_split,
)
from training.utils import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "training" / "config.yaml"
DEFAULT_ARTIFACTS = REPO_ROOT / "training" / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--recompute-statistics",
        action="store_true",
        help="Fit normalization only on Normal Train and write it to --artifacts-dir.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_statistics(
    config: dict,
    train_rows: pd.DataFrame,
    artifacts_dir: Path,
    overwrite: bool,
) -> None:
    destinations = [
        artifacts_dir / "audio_normalization.json",
        artifacts_dir / "imu_normalization.json",
        artifacts_dir / "mel_filterbank.npy",
    ]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to replace preprocessing artifacts; first existing file: {existing[0]}"
        )
    audio_stats, imu_stats, filterbank = compute_train_statistics(config, train_rows)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in zip(destinations[:2], (audio_stats, imu_stats)):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
    np.save(destinations[2], filterbank)


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    metadata = load_metadata(config)
    audit = audit_splits(config, metadata)
    cache_root = args.cache_dir.resolve() if args.cache_dir else Path(config["paths"]["cache_dir"])
    artifacts_dir = args.artifacts_dir.resolve()
    train_rows = rows_for_split(config, metadata, "train")
    if args.recompute_statistics:
        write_statistics(config, train_rows, artifacts_dir, args.overwrite)

    required = [
        artifacts_dir / "audio_normalization.json",
        artifacts_dir / "imu_normalization.json",
        artifacts_dir / "mel_filterbank.npy",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessing artifact: {missing[0]}")

    manifest: list[dict[str, str]] = []
    for split in config["data"]["splits"]:
        rows = rows_for_split(config, metadata, split)
        for _, row in rows.iterrows():
            destination = cache_path(cache_root, str(row["dataset"]), str(row["sample_id"]))
            if args.overwrite or not destination.is_file():
                values = preprocess_row(config, row, artifacts_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **values)
            manifest.append(
                {
                    "split": split,
                    "dataset": str(row["dataset"]),
                    "sample_id": str(row["sample_id"]),
                    "label": str(row["label_type"]),
                    "cache_file": str(destination.relative_to(cache_root)).replace("\\", "/"),
                }
            )
        print(f"Cached {len(rows)} {split} samples.")
    cache_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(cache_root / "cache_manifest.csv", index=False)
    with (cache_root / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=True, allow_nan=False)
    print(f"Preprocessing cache is ready at {cache_root}")


if __name__ == "__main__":
    main()
