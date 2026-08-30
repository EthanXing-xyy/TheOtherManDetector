"""Fitted K=2 shared-covariance Mahalanobis anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


EMBEDDING_DIMENSION = 32
PROTOTYPE_COUNT = 2
BASE_ANOMALY_THRESHOLD = 5.692456722
ANOMALY_THRESHOLD_SAFETY_FACTOR = 1.05
DEPLOYMENT_ANOMALY_THRESHOLD = (
    BASE_ANOMALY_THRESHOLD * ANOMALY_THRESHOLD_SAFETY_FACTOR
)


@dataclass(frozen=True)
class MahalanobisDetector:
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    embedding_std: np.ndarray
    prototypes: np.ndarray
    precision: np.ndarray
    anomaly_threshold: np.float32

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisDetector":
        with np.load(Path(path), allow_pickle=False) as archive:
            expected = {
                "embedding_mean",
                "embedding_scale",
                "embedding_std",
                "prototypes",
                "precision",
                "anomaly_threshold",
            }
            if set(archive.files) != expected:
                raise ValueError(f"Unexpected detector keys: {archive.files}")
            detector = cls(
                embedding_mean=np.asarray(archive["embedding_mean"], dtype=np.float32),
                embedding_scale=np.asarray(archive["embedding_scale"], dtype=np.float32),
                embedding_std=np.asarray(archive["embedding_std"], dtype=np.float32),
                prototypes=np.asarray(archive["prototypes"], dtype=np.float32),
                precision=np.asarray(archive["precision"], dtype=np.float32),
                anomaly_threshold=np.float32(archive["anomaly_threshold"]),
            )
        detector.validate()
        return detector

    def validate(self) -> None:
        expected_shapes = {
            "embedding_mean": (EMBEDDING_DIMENSION,),
            "embedding_scale": (EMBEDDING_DIMENSION,),
            "embedding_std": (EMBEDDING_DIMENSION,),
            "prototypes": (PROTOTYPE_COUNT, EMBEDDING_DIMENSION),
            "precision": (EMBEDDING_DIMENSION, EMBEDDING_DIMENSION),
        }
        for name, shape in expected_shapes.items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid detector parameter {name}: {value.shape}")
        if np.any(self.embedding_scale <= 0) or not np.isfinite(self.anomaly_threshold):
            raise ValueError("Detector scaling or threshold is invalid")

    def distances(self, embeddings: np.ndarray) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(f"Expected N x 32 embeddings, got {matrix.shape}")
        standardized = (matrix - self.embedding_mean) / self.embedding_scale
        differences = standardized[:, None, :] - self.prototypes[None, :, :]
        squared = np.einsum(
            "nki,ij,nkj->nk",
            differences,
            self.precision,
            differences,
            optimize=True,
        )
        return np.sqrt(np.maximum(squared, 0.0)).astype(np.float32)

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        return np.min(self.distances(embeddings), axis=1)

    def predict(
        self, embeddings: np.ndarray, *, deployment_safety_factor: bool = False
    ) -> np.ndarray:
        threshold = float(self.anomaly_threshold)
        if deployment_safety_factor:
            threshold *= ANOMALY_THRESHOLD_SAFETY_FACTOR
        return self.scores(embeddings) > threshold


def score_c_order(
    embedding: np.ndarray, detector: MahalanobisDetector
) -> tuple[np.float32, int, np.ndarray]:
    """Match the scalar float32 accumulation order used by the firmware."""

    value = np.asarray(embedding, dtype=np.float32)
    if value.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(f"Expected one 32-D embedding, got {value.shape}")
    standardized = np.asarray(
        (value - detector.embedding_mean) / detector.embedding_scale,
        dtype=np.float32,
    )
    distances = np.empty(PROTOTYPE_COUNT, dtype=np.float32)
    for prototype_index in range(PROTOTYPE_COUNT):
        difference = np.asarray(
            standardized - detector.prototypes[prototype_index], dtype=np.float32
        )
        squared = np.float32(0.0)
        for row in range(EMBEDDING_DIMENSION):
            row_product = np.float32(0.0)
            for column in range(EMBEDDING_DIMENSION):
                row_product = np.float32(
                    row_product
                    + np.float32(detector.precision[row, column] * difference[column])
                )
            squared = np.float32(squared + np.float32(difference[row] * row_product))
        distances[prototype_index] = np.float32(
            np.sqrt(np.float32(max(float(squared), 0.0)))
        )
    nearest = 1 if distances[1] < distances[0] else 0
    return distances[nearest], nearest, distances
