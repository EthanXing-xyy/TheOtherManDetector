#!/usr/bin/env python
"""Fit the K=2 Mahalanobis anomaly head from Normal Train embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from training.anomaly_head import MahalanobisDetector, score_c_order
from training.data import CachedMultimodalDataset, load_metadata, rows_for_split
from training.models.deployment_model import load_checkpoint_encoder
from training.utils import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "training" / "config.yaml"
DEFAULT_CHECKPOINT = REPO_ROOT / "training" / "checkpoints" / "reconstruction_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "training" / "outputs" / "mahalanobis_detector.npz",
    )
    return parser.parse_args()


def extract_embeddings(
    model: torch.nn.Module, dataset: CachedMultimodalDataset
) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            output = model(sample["audio"].unsqueeze(0), sample["imu"].unsqueeze(0))
            if tuple(output.shape) != (1, 32) or not torch.isfinite(output).all():
                raise ValueError("Embedding model returned an invalid output")
            embeddings.append(output.numpy()[0].copy())
    return np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    metadata = load_metadata(config)
    cache_root = args.cache_dir.resolve() if args.cache_dir else Path(config["paths"]["cache_dir"])
    train_dataset = CachedMultimodalDataset(
        cache_root, rows_for_split(config, metadata, "train")
    )
    validation_dataset = CachedMultimodalDataset(
        cache_root, rows_for_split(config, metadata, "validation")
    )
    model = load_checkpoint_encoder(args.checkpoint, converter_compatible=True)
    train_embeddings = extract_embeddings(model, train_dataset)
    validation_embeddings = extract_embeddings(model, validation_dataset)
    if train_embeddings.shape != (123, 32) or validation_embeddings.shape != (16, 32):
        raise AssertionError("Anomaly fitting requires 123 training and 16 validation embeddings")

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    standardized = np.asarray(scaler.fit_transform(train_embeddings), dtype=np.float64)
    kmeans = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=50,
        max_iter=1000,
        random_state=2026,
    )
    assignments = np.asarray(kmeans.fit_predict(standardized), dtype=np.int64)
    prototypes = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    cluster_sizes = np.bincount(assignments, minlength=2)
    if np.any(cluster_sizes == 0):
        raise AssertionError("KMeans produced an empty normal prototype")
    residuals = standardized - prototypes[assignments]
    covariance = LedoitWolf(assume_centered=True, store_precision=True).fit(residuals)
    detector = MahalanobisDetector(
        embedding_mean=np.asarray(scaler.mean_, dtype=np.float32),
        embedding_scale=np.asarray(scaler.scale_, dtype=np.float32),
        embedding_std=np.asarray(np.sqrt(scaler.var_), dtype=np.float32),
        prototypes=np.asarray(prototypes, dtype=np.float32),
        precision=np.asarray(covariance.precision_, dtype=np.float32),
        anomaly_threshold=np.float32(0.0),
    )
    detector.validate()
    validation_scores = np.asarray(
        [score_c_order(value, detector)[0] for value in validation_embeddings],
        dtype=np.float32,
    )
    detector = replace(detector, anomaly_threshold=np.float32(validation_scores.max()))
    detector.validate()
    if np.count_nonzero(validation_scores > detector.anomaly_threshold):
        raise AssertionError("Validation-maximum threshold produced a validation false positive")

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        embedding_mean=detector.embedding_mean,
        embedding_scale=detector.embedding_scale,
        embedding_std=detector.embedding_std,
        prototypes=detector.prototypes,
        precision=detector.precision,
        anomaly_threshold=detector.anomaly_threshold,
    )
    summary = {
        "embedding_source": "converter-compatible encoder evaluated one sample at a time",
        "fit_data": "123 Normal Train samples only",
        "threshold_data": "maximum score over 16 Normal Validation samples",
        "kmeans": {
            "clusters": 2,
            "initialization": "k-means++",
            "n_init": 50,
            "max_iter": 1000,
            "random_state": 2026,
            "cluster_sizes": cluster_sizes.astype(int).tolist(),
            "inertia": float(kmeans.inertia_),
            "silhouette": float(silhouette_score(standardized, assignments)),
        },
        "covariance": "shared LedoitWolf precision fitted to within-prototype residuals",
        "base_anomaly_threshold": float(detector.anomaly_threshold),
    }
    with destination.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True, allow_nan=False)
    print(f"Wrote detector parameters to {destination}")


if __name__ == "__main__":
    main()
