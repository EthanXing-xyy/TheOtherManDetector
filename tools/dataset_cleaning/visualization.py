"""Diagnostic plots. Plot-only summaries never replace or alter saved raw data."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset_io import AudioRecording, ImuRecording, LabelInterval
from .onset_detection import OnsetResult


BOUNDARY_COLORS = {
    "label": "#222222",
    "crop_start": "#2563eb",
    "onset": "#dc2626",
    "crop_end": "#16a34a",
}


def _audio_rms_inside_label(
    audio: AudioRecording,
    audio_offset: float,
    label_start: float,
    label_end: float,
    origin: float,
) -> tuple[np.ndarray, np.ndarray]:
    rate = audio.sample_rate
    first = max(0, int(math.ceil((label_start - audio_offset) * rate - 1e-12)))
    stop = min(audio.frame_count, int(math.ceil((label_end - audio_offset) * rate - 1e-12)))
    if stop <= first:
        return np.empty(0), np.empty(0)
    values = audio.samples[first:stop].astype(np.float64)
    block = max(1, int(round(0.010 * rate)))
    starts = np.arange(0, values.size, block, dtype=np.int64)
    squared_sums = np.add.reduceat(values * values, starts)
    counts = np.diff(np.append(starts, values.size))
    rms = np.sqrt(squared_sums / counts)
    centers = starts + counts / 2.0
    times = audio_offset + (first + centers) / rate - origin
    return times, rms


def _draw_boundaries(
    axis: plt.Axes,
    label_start_rel: float,
    label_end_rel: float,
    crop_start_rel: float | None,
    crop_end_rel: float | None,
    onset_available: bool,
    with_labels: bool = False,
) -> None:
    axis.axvline(
        label_start_rel,
        color=BOUNDARY_COLORS["label"],
        linestyle=":",
        linewidth=1.3,
        label="label start" if with_labels else None,
    )
    if crop_start_rel is not None:
        axis.axvline(
            crop_start_rel,
            color=BOUNDARY_COLORS["crop_start"],
            linestyle="--",
            linewidth=1.3,
            label="crop start (-2.5 s)" if with_labels else None,
        )
    if onset_available:
        axis.axvline(
            0.0,
            color=BOUNDARY_COLORS["onset"],
            linestyle="-",
            linewidth=1.5,
            label="door onset (0 s)" if with_labels else None,
        )
    if crop_end_rel is not None:
        axis.axvline(
            crop_end_rel,
            color=BOUNDARY_COLORS["crop_end"],
            linestyle="--",
            linewidth=1.3,
            label="crop end (+4.7 s)" if with_labels else None,
        )
    axis.axvline(
        label_end_rel,
        color=BOUNDARY_COLORS["label"],
        linestyle=":",
        linewidth=1.3,
        label="label end" if with_labels else None,
    )


def plot_sample_diagnostic(
    output_path: Path,
    dataset: str,
    label: LabelInterval,
    imu: ImuRecording,
    audio: AudioRecording,
    audio_offset: float,
    onset: OnsetResult,
    status: str,
    reason: str,
    crop_pre_seconds: float,
    crop_post_seconds: float,
) -> None:
    candidate_time = onset.candidate_time
    origin = candidate_time if candidate_time is not None else label.start
    label_start_rel = label.start - origin
    label_end_rel = label.end - origin
    crop_start_rel = -crop_pre_seconds if candidate_time is not None else None
    crop_end_rel = crop_post_seconds if candidate_time is not None else None

    imu_times = onset.label_timestamps - origin
    imu_values = imu.values[onset.label_indices] if onset.label_indices.size else np.empty((0, 6))
    audio_times, audio_rms = _audio_rms_inside_label(
        audio, audio_offset, label.start, label.end, origin
    )

    left_candidates = [label_start_rel]
    right_candidates = [label_end_rel]
    if crop_start_rel is not None:
        left_candidates.append(crop_start_rel)
    if crop_end_rel is not None:
        right_candidates.append(crop_end_rel)
    left = min(left_candidates) - 0.10
    right = max(right_candidates) + 0.10

    figure, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    accel_colors = ("#1d4ed8", "#059669", "#d97706")
    gyro_colors = ("#7c3aed", "#db2777", "#0891b2")
    for axis_index, (name, color) in enumerate(zip(("ax", "ay", "az"), accel_colors)):
        if imu_values.size:
            axes[0].plot(imu_times, imu_values[:, axis_index], label=name, color=color, linewidth=0.8)
    axes[0].set_ylabel("Accel (raw)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    for axis_index, (name, color) in enumerate(zip(("gx", "gy", "gz"), gyro_colors), start=3):
        if imu_values.size:
            axes[1].plot(imu_times, imu_values[:, axis_index], label=name, color=color, linewidth=0.8)
    axes[1].set_ylabel("Gyro (raw)")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)

    if onset.motion_score.size:
        axes[2].plot(imu_times, onset.motion_score, color="#111827", linewidth=1.0, label="motion score")
    if math.isfinite(onset.threshold):
        axes[2].axhline(
            onset.threshold,
            color="#f97316",
            linestyle="--",
            linewidth=1.2,
            label="onset/departure threshold",
        )
    if (
        math.isfinite(onset.confirmation_threshold)
        and abs(onset.confirmation_threshold - onset.threshold) > 1e-12
    ):
        axes[2].axhline(
            onset.confirmation_threshold,
            color="#9333ea",
            linestyle=":",
            linewidth=1.2,
            label="confirmation threshold",
        )
    if onset.confirmation_time is not None and candidate_time is not None:
        axes[2].axvline(
            onset.confirmation_time - candidate_time,
            color="#9333ea",
            linestyle=":",
            linewidth=1.0,
            label="persistent-event confirmation",
        )
    axes[2].set_ylabel("IMU score")
    axes[2].legend(loc="upper right", fontsize=8)

    if audio_times.size:
        axes[3].plot(audio_times, audio_rms, color="#475569", linewidth=0.8, label="10 ms audio RMS")
    axes[3].set_ylabel("Audio RMS\n(int16 units)")
    axes[3].set_xlabel("Time relative to detected door onset (s)")
    axes[3].legend(loc="upper right", fontsize=8)

    for index, axis in enumerate(axes):
        _draw_boundaries(
            axis,
            label_start_rel,
            label_end_rel,
            crop_start_rel,
            crop_end_rel,
            candidate_time is not None,
            with_labels=index == 0,
        )
        axis.grid(True, alpha=0.20)
        axis.set_xlim(left, right)
        if left < label_start_rel:
            axis.axvspan(left, label_start_rel, color="#ef4444", alpha=0.05)
        if right > label_end_rel:
            axis.axvspan(label_end_rel, right, color="#ef4444", alpha=0.05)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right", ncol=4, fontsize=7)

    candidate_text = "none" if candidate_time is None else f"{candidate_time:.7f} s absolute"
    score_text = (
        "n/a"
        if onset.onset_score is None
        else f"{onset.onset_score:.3f} / departure threshold {onset.threshold:.3f}, confidence {onset.confidence_ratio:.3f}"
    )
    confirmation_text = (
        "n/a"
        if onset.confirmation_time is None
        else f"{onset.confirmation_time:.7f} s (backtrack {onset.backtrack_seconds:.3f} s)"
    )
    figure.suptitle(
        f"{dataset} / sample_{label.source_index:03d} — {label.label} — {status}\n"
        f"t0={candidate_text}; onset score={score_text}; confirmation={confirmation_text}",
        fontsize=12,
    )
    footer = (
        "Entire human-labelled interval is shown. Motion score/RMS are diagnostic-only; "
        "saved IMU and WAV remain raw."
    )
    if reason:
        footer += f"  Reason: {reason}"
    figure.text(0.01, 0.008, footer, fontsize=8)
    figure.tight_layout(rect=(0, 0.025, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=130)
    plt.close(figure)


def plot_overview(
    output_path: Path,
    title: str,
    relative_times: np.ndarray,
    traces: list[np.ndarray],
    trace_names: list[str],
) -> None:
    figure, axis = plt.subplots(figsize=(14, 7))
    if traces:
        matrix = np.vstack(traces)
        for trace in matrix:
            axis.plot(relative_times, trace, color="#64748b", alpha=0.16, linewidth=0.65)
        median = np.median(matrix, axis=0)
        lower = np.percentile(matrix, 5, axis=0)
        upper = np.percentile(matrix, 95, axis=0)
        axis.fill_between(relative_times, lower, upper, color="#60a5fa", alpha=0.24, label="5th–95th percentile")
        axis.plot(relative_times, median, color="#1d4ed8", linewidth=2.2, label="median")
    axis.axvline(-2.5, color=BOUNDARY_COLORS["crop_start"], linestyle="--", label="crop start")
    axis.axvline(0.0, color=BOUNDARY_COLORS["onset"], linewidth=1.5, label="door onset")
    axis.axvline(4.7, color=BOUNDARY_COLORS["crop_end"], linestyle="--", label="crop end")
    axis.set_xlim(-2.5, 4.7)
    axis.set_xlabel("Time relative to door onset (s)")
    axis.set_ylabel("Normalized IMU motion score (diagnostic only)")
    axis.set_title(f"{title} — onset aligned successful samples (n={len(traces)})")
    axis.grid(True, alpha=0.22)
    axis.legend(loc="upper right")
    note = "Individual traces + median + 5th–95th percentile; no time warping or resampling."
    if trace_names:
        note += f"  Sources: {len(trace_names)} paired samples."
    figure.text(0.01, 0.01, note, fontsize=8)
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_error_diagnostic(
    output_path: Path,
    dataset: str,
    label: LabelInterval,
    reason: str,
) -> None:
    """Create a visible per-label placeholder when source/plot processing fails."""
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.axvline(0.0, color=BOUNDARY_COLORS["label"], linestyle=":", label="label start")
    axis.axvline(
        label.duration,
        color=BOUNDARY_COLORS["label"],
        linestyle=":",
        label="label end",
    )
    axis.set_xlim(-0.1, label.duration + 0.1)
    axis.set_ylim(-1.0, 1.0)
    axis.set_xlabel("Time relative to label start (s); t0 unavailable")
    axis.set_yticks([])
    axis.grid(True, alpha=0.2)
    axis.legend(loc="upper right")
    axis.text(
        0.5,
        0.5,
        f"ERROR diagnostic\n{reason}",
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="#b91c1c",
        fontsize=11,
        wrap=True,
    )
    axis.set_title(
        f"{dataset} / sample_{label.source_index:03d} — {label.label} — ERROR"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
