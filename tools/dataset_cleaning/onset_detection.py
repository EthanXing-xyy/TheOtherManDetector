"""Adaptive IMU-only door-onset detection inside a human-labelled interval."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class OnsetResult:
    label_indices: np.ndarray
    label_timestamps: np.ndarray
    motion_score: np.ndarray
    threshold: float
    confirmation_threshold: float
    baseline_duration: float
    baseline_score_median: float
    noise_scales: np.ndarray
    candidate_local_index: int | None
    candidate_global_index: int | None
    candidate_time: float | None
    onset_score: float | None
    confirmation_local_index: int | None
    confirmation_global_index: int | None
    confirmation_time: float | None
    confirmation_score: float | None
    backtrack_seconds: float | None
    local_peak_score: float | None
    confidence_ratio: float | None
    uncertain: bool
    reason: str


def _minimum_positive_step(values: np.ndarray) -> float:
    unique = np.unique(values)
    if unique.size < 2:
        return 0.0
    steps = np.diff(unique)
    positive = steps[steps > np.finfo(np.float64).eps]
    return float(np.min(positive)) if positive.size else 0.0


def _trailing_rms_across_axes(z_scores: np.ndarray, window: int) -> np.ndarray:
    squared = z_scores * z_scores
    cumulative = np.vstack(
        (np.zeros((1, squared.shape[1]), dtype=np.float64), np.cumsum(squared, axis=0))
    )
    output = np.empty(squared.shape[0], dtype=np.float64)
    for index in range(squared.shape[0]):
        lower = max(0, index - window + 1)
        mean_per_axis = (cumulative[index + 1] - cumulative[lower]) / (index - lower + 1)
        output[index] = float(np.sqrt(np.mean(mean_per_axis)))
    return output


def detect_door_onset(
    timestamps: np.ndarray,
    values: np.ndarray,
    label_start: float,
    label_end: float,
    sample_rate: float,
) -> OnsetResult:
    """
    Detect the first persistent motion inside the label only.

    Saved signals are never filtered. Smoothing here is a temporary causal RMS of
    per-axis first differences, normalized by the label's initial quiet baseline.
    """
    tolerance = 1e-10
    label_indices = np.flatnonzero(
        (timestamps >= label_start - tolerance) & (timestamps < label_end - tolerance)
    )
    if label_indices.size < max(20, int(round(sample_rate * 0.5))):
        return OnsetResult(
            label_indices=label_indices,
            label_timestamps=timestamps[label_indices],
            motion_score=np.zeros(label_indices.size, dtype=np.float64),
            threshold=float("nan"),
            confirmation_threshold=float("nan"),
            baseline_duration=0.0,
            baseline_score_median=float("nan"),
            noise_scales=np.full(6, float("nan")),
            candidate_local_index=None,
            candidate_global_index=None,
            candidate_time=None,
            onset_score=None,
            confirmation_local_index=None,
            confirmation_global_index=None,
            confirmation_time=None,
            confirmation_score=None,
            backtrack_seconds=None,
            local_peak_score=None,
            confidence_ratio=None,
            uncertain=True,
            reason="Too few IMU samples inside the labelled interval",
        )

    label_times = timestamps[label_indices]
    label_values = values[label_indices]
    duration = label_end - label_start
    baseline_duration = min(1.25, max(0.50, duration * 0.12))
    baseline_mask = label_times < label_start + baseline_duration

    differences = np.zeros_like(label_values)
    differences[1:] = np.diff(label_values, axis=0)
    baseline_differences = differences[baseline_mask]
    if baseline_differences.shape[0] < 10:
        baseline_differences = differences[: min(differences.shape[0], 50)]

    centers = np.median(baseline_differences, axis=0)
    absolute_centered = np.abs(baseline_differences - centers)
    mad_sigma = 1.4826 * np.median(absolute_centered, axis=0)
    q90_sigma = np.quantile(absolute_centered, 0.90, axis=0) / 1.6448536269514722
    half_std = 0.5 * np.std(baseline_differences, axis=0)
    quantization_steps = np.asarray(
        [_minimum_positive_step(label_values[:, axis]) for axis in range(6)],
        dtype=np.float64,
    )
    numerical_floor = np.full(6, np.finfo(np.float64).eps)
    noise_scales = np.maximum.reduce(
        (mad_sigma, q90_sigma, half_std, quantization_steps, numerical_floor)
    )

    normalized = np.clip((differences - centers) / noise_scales, -30.0, 30.0)
    smoothing_samples = max(3, int(round(0.10 * sample_rate)))
    score = _trailing_rms_across_axes(normalized, smoothing_samples)

    baseline_score_mask = baseline_mask.copy()
    baseline_score_mask[:smoothing_samples] = False
    baseline_scores = score[baseline_score_mask]
    if baseline_scores.size < 10:
        baseline_scores = score[smoothing_samples : max(smoothing_samples + 10, int(0.5 * sample_rate))]
    baseline_median = float(np.median(baseline_scores))
    baseline_mad = float(1.4826 * np.median(np.abs(baseline_scores - baseline_median)))
    baseline_extreme = float(np.quantile(baseline_scores, 0.995))
    # 4 is a dimensionless normalized-score guard, not a fixed sensor-unit threshold.
    confirmation_threshold = max(
        baseline_median + 8.0 * baseline_mad, 1.35 * baseline_extreme, 4.0
    )

    search_start = int(np.searchsorted(label_times, label_start + baseline_duration, side="left"))
    above = score > confirmation_threshold
    persistence_window = max(3, int(round(0.05 * sample_rate)))
    required_above = max(3, persistence_window - 1)
    candidate_local: int | None = None
    last_start = max(search_start, score.size - persistence_window + 1)
    for index in range(search_start, last_start):
        if above[index] and int(np.count_nonzero(above[index : index + persistence_window])) >= required_above:
            candidate_local = index
            break

    if candidate_local is None:
        if search_start < score.size:
            candidate_local = search_start + int(np.argmax(score[search_start:]))
            candidate_global = int(label_indices[candidate_local])
            candidate_time = float(label_times[candidate_local])
            candidate_score = float(score[candidate_local])
            ratio = (
                candidate_score / confirmation_threshold
                if confirmation_threshold > 0
                else float("nan")
            )
        else:
            candidate_global = None
            candidate_time = None
            candidate_score = None
            ratio = None
        return OnsetResult(
            label_indices=label_indices,
            label_timestamps=label_times,
            motion_score=score,
            threshold=confirmation_threshold,
            confirmation_threshold=confirmation_threshold,
            baseline_duration=baseline_duration,
            baseline_score_median=baseline_median,
            noise_scales=noise_scales,
            candidate_local_index=candidate_local,
            candidate_global_index=candidate_global,
            candidate_time=candidate_time,
            onset_score=candidate_score,
            confirmation_local_index=None,
            confirmation_global_index=None,
            confirmation_time=None,
            confirmation_score=None,
            backtrack_seconds=None,
            local_peak_score=candidate_score,
            confidence_ratio=ratio,
            uncertain=True,
            reason="No persistent adaptive-threshold crossing inside label",
        )

    confirmation_local = candidate_local
    confirmation_global = int(label_indices[confirmation_local])
    confirmation_time = float(label_times[confirmation_local])
    confirmation_score = float(score[confirmation_local])

    # The high threshold confirms that a real, persistent event exists. Door
    # Slight anomalous motion can begin as a gradual rise below that
    # confirmation threshold, so backtrack with a lower adaptive hysteresis
    # threshold to the first baseline departure continuously connected to the
    # confirmed event. This still never examines data outside the label.
    threshold = max(
        baseline_median + 3.0 * baseline_mad,
        1.10 * baseline_extreme,
        baseline_median + 0.30 * (confirmation_threshold - baseline_median),
    )
    backtrack_limit = max(1, int(round(1.25 * sample_rate)))
    backtrack_start = max(search_start, confirmation_local - backtrack_limit)
    departure_above = score > threshold
    departure_window = max(5, int(round(0.10 * sample_rate)))
    departure_required = max(4, int(math.ceil(0.60 * departure_window)))
    # Slow openings can contain a brief pause after the first real displacement.
    # Keep such motion connected for up to 0.35 s; the overall 1.25 s backtrack
    # cap and persistent departure test prevent linking distant noise events.
    maximum_quiet_gap = max(3, int(round(0.35 * sample_rate)))

    def longest_false_run(mask: np.ndarray) -> int:
        longest = 0
        current = 0
        for value in mask:
            if value:
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return longest

    candidate_local = confirmation_local
    for index in range(backtrack_start, confirmation_local + 1):
        persistence_stop = min(score.size, index + departure_window)
        persistent = (
            departure_above[index]
            and int(np.count_nonzero(departure_above[index:persistence_stop]))
            >= departure_required
        )
        connected = (
            longest_false_run(departure_above[index : confirmation_local + 1])
            <= maximum_quiet_gap
        )
        if persistent and connected:
            candidate_local = index
            break

    candidate_global = int(label_indices[candidate_local])
    candidate_time = float(label_times[candidate_local])
    onset_score = float(score[candidate_local])
    backtrack_seconds = confirmation_time - candidate_time
    peak_end = min(
        score.size,
        confirmation_local + max(1, int(round(0.40 * sample_rate))),
    )
    local_peak_score = float(np.max(score[confirmation_local:peak_end]))
    confidence_ratio = local_peak_score / confirmation_threshold

    uncertainty_reasons: list[str] = []
    # A 4-of-5 persistent crossing already gives strong evidence. The additional
    # local-peak guard is deliberately modest so very gentle openings are not
    # rejected merely for having low amplitude.
    if confidence_ratio < 1.15:
        uncertainty_reasons.append(
            f"local peak/threshold ratio {confidence_ratio:.3f} is below 1.15"
        )
    if candidate_time < label_start + baseline_duration - 1e-9:
        uncertainty_reasons.append("candidate overlaps baseline estimation interval")

    return OnsetResult(
        label_indices=label_indices,
        label_timestamps=label_times,
        motion_score=score,
        threshold=threshold,
        confirmation_threshold=confirmation_threshold,
        baseline_duration=baseline_duration,
        baseline_score_median=baseline_median,
        noise_scales=noise_scales,
        candidate_local_index=candidate_local,
        candidate_global_index=candidate_global,
        candidate_time=candidate_time,
        onset_score=onset_score,
        confirmation_local_index=confirmation_local,
        confirmation_global_index=confirmation_global,
        confirmation_time=confirmation_time,
        confirmation_score=confirmation_score,
        backtrack_seconds=backtrack_seconds,
        local_peak_score=local_peak_score,
        confidence_ratio=confidence_ratio,
        uncertain=bool(uncertainty_reasons),
        reason="; ".join(uncertainty_reasons),
    )
