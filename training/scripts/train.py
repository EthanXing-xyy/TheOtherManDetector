#!/usr/bin/env python
"""Train the reconstruction model using Normal Train and Normal Validation.

Author: Ethan_Xing
License: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from training.data import (
    CachedMultimodalDataset,
    audit_splits,
    load_metadata,
    rows_for_split,
)
from training.losses import reconstruction_losses
from training.models.components import parameter_counts
from training.models.training_model import TrainingModel
from training.utils import configure_cpu_threads, load_config, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "training" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def make_loader(
    dataset: CachedMultimodalDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )


def run_epoch(
    model: TrainingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> tuple[dict[str, float], float | None]:
    training = optimizer is not None
    model.train(training)
    totals = {"joint": 0.0, "audio": 0.0, "imu": 0.0}
    sample_count = 0
    maximum_gradient: float | None = None
    for batch in loader:
        audio = batch["audio"]
        imu = batch["imu"]
        target = batch["target"]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction = model(audio, imu)
            losses = reconstruction_losses(reconstruction, target)
            if not torch.isfinite(losses["joint"]):
                raise FloatingPointError("Joint reconstruction loss is not finite")
            if training:
                losses["joint"].backward()
                norm = clip_grad_norm_(model.parameters(), gradient_clip_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Gradient norm is not finite")
                maximum_gradient = max(float(norm), maximum_gradient or 0.0)
                optimizer.step()
        count = int(audio.shape[0])
        for name in totals:
            totals[name] += float(losses[name].detach()) * count
        sample_count += count
    if sample_count != len(loader.dataset):
        raise AssertionError("An epoch did not consume every dataset sample exactly once")
    averages = {name: value / sample_count for name, value in totals.items()}
    expected_joint = (40.0 * averages["audio"] + 24.0 * averages["imu"]) / 64.0
    if not np.isclose(averages["joint"], expected_joint, rtol=2e-6, atol=1e-8):
        raise AssertionError("Joint reconstruction loss is inconsistent with its diagnostics")
    return averages, maximum_gradient


def train_seed(
    config: dict[str, Any],
    train_dataset: CachedMultimodalDataset,
    validation_dataset: CachedMultimodalDataset,
    seed: int,
    output_dir: Path,
) -> tuple[Path, pd.DataFrame, float]:
    set_seed(seed)
    model = TrainingModel(config).cpu()
    counts = parameter_counts(model)
    if counts["total_parameters"] != 77904:
        raise AssertionError(f"Architecture parameter count changed: {counts}")
    training_config = config["training"]
    train_loader = make_loader(
        train_dataset,
        int(training_config["batch_size"]),
        True,
        seed,
        int(training_config["num_workers"]),
    )
    validation_loader = make_loader(
        validation_dataset,
        int(training_config["batch_size"]),
        False,
        seed,
        int(training_config["num_workers"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config["scheduler_factor"]),
        patience=int(training_config["scheduler_patience"]),
        min_lr=float(training_config["scheduler_min_lr"]),
    )
    history_rows: list[dict[str, float | int]] = []
    best_validation = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "checkpoints" / f"training_run_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, int(training_config["max_epochs"]) + 1):
        train_losses, maximum_gradient = run_epoch(
            model,
            train_loader,
            optimizer,
            float(training_config["gradient_clip_norm"]),
        )
        with torch.inference_mode():
            validation_losses, _ = run_epoch(
                model,
                validation_loader,
                None,
                float(training_config["gradient_clip_norm"]),
            )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "maximum_preclip_gradient_norm": float(maximum_gradient or 0.0),
                **{f"train_{name}": value for name, value in train_losses.items()},
                **{f"validation_{name}": value for name, value in validation_losses.items()},
            }
        )
        improved = validation_losses["joint"] < best_validation
        if improved:
            best_validation = validation_losses["joint"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": "TrainingModel",
                    "feature_attention": "directed audio-IMU bipartite attention",
                    "temporal_grid": "180 x 40 ms local frames; 90 x 80 ms global tokens",
                    "seed": seed,
                    "epoch": epoch,
                    "best_validation_joint": float(best_validation),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "parameter_counts": counts,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        scheduler.step(validation_losses["joint"])
        marker = " best" if improved else ""
        print(
            f"run={seed} epoch={epoch:03d} train={train_losses['joint']:.6f} "
            f"validation={validation_losses['joint']:.6f} lr={learning_rate:.2e}{marker}",
            flush=True,
        )
        if epochs_without_improvement >= int(training_config["early_stopping_patience"]):
            break
    history = pd.DataFrame(history_rows)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_dir / f"training_run_{seed}.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["training_seconds"] = float(time.perf_counter() - started)
    checkpoint["best_epoch"] = best_epoch
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, history, float(best_validation)


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    if str(config["experiment"].get("device", "cpu")).lower() != "cpu":
        raise ValueError("The recorded deterministic training protocol is CPU-only")
    configure_cpu_threads(config)
    metadata = load_metadata(config)
    audit = audit_splits(config, metadata)
    cache_root = args.cache_dir.resolve() if args.cache_dir else Path(config["paths"]["cache_dir"])
    output_dir = args.output_dir.resolve() if args.output_dir else Path(config["paths"]["output_dir"])
    train_dataset = CachedMultimodalDataset(cache_root, rows_for_split(config, metadata, "train"))
    validation_dataset = CachedMultimodalDataset(
        cache_root, rows_for_split(config, metadata, "validation")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in (int(value) for value in config["experiment"]["seeds"]):
        path, history, best = train_seed(
            config, train_dataset, validation_dataset, seed, output_dir
        )
        summaries.append(
            {
                "run_seed": seed,
                "checkpoint": str(path.relative_to(output_dir)).replace("\\", "/"),
                "epochs": int(len(history)),
                "best_validation_joint": best,
            }
        )
    selected = min(summaries, key=lambda row: (row["best_validation_joint"], row["run_seed"]))
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection": "minimum Normal Validation joint reconstruction MSE",
                "abnormal_samples_used": False,
                "dataset_audit": audit,
                "runs": summaries,
                "selected": selected,
            },
            handle,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    print(f"Selected {selected['checkpoint']} using Normal Validation only.")


if __name__ == "__main__":
    main()
