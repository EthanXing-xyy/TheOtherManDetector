"""Crop Audio + 6-axis IMU samples to the fixed [-2.5, +4.7) s window."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

from tools.dataset_cleaning.dataset_io import (
    AudioRecording,
    ImuRecording,
    LabelInterval,
    SessionInfo,
    discover_sessions,
    ensure_safe_roots,
    ensure_safe_output_path,
    hash_source_tree,
    read_audio,
    read_imu,
    read_labels,
    remove_previous_generated_files,
    verify_written_audio,
    write_audio_crop,
    write_generated_manifest,
    write_imu_crop,
    write_json,
)
from tools.dataset_cleaning.inspection import inspect_database
from tools.dataset_cleaning.onset_detection import OnsetResult, detect_door_onset
from tools.dataset_cleaning.visualization import (
    plot_error_diagnostic,
    plot_overview,
    plot_sample_diagnostic,
)


DEFAULT_INPUT = REPO_ROOT / "dataset" / "raw"
DEFAULT_OUTPUT = REPO_ROOT / "training" / "outputs" / "processed_dataset"
PRE_SECONDS = 2.5
POST_SECONDS = 4.7
WINDOW_SECONDS = PRE_SECONDS + POST_SECONDS
DATASET_ORDER = (
    "normal_train_01",
    "normal_train_02",
    "normal_train_03",
    "normal_train_04",
    "normal_train_05",
    "normal_train_06",
    "normal_train_07",
    "normal_train_08",
    "normal_validation",
    "normal_test",
    "fierce_anomaly",
    "slight_anomaly",
)

LABEL_NAMES = {
    "NORMAL": "NORMAL",
    "AbnormalSlight": "SLIGHT_ANOMALY",
    "AbnormalFierce": "FIERCE_ANOMALY",
}

METADATA_FIELDS = (
    "dataset",
    "sample_id",
    "label_type",
    "label_confidence",
    "label_comment",
    "original_label_start",
    "original_label_end",
    "original_label_duration",
    "detected_t0_absolute",
    "t0_relative_to_label_start",
    "crop_start_absolute",
    "crop_end_absolute",
    "pre_available",
    "post_available",
    "imu_sample_count",
    "audio_sample_count",
    "imu_sample_rate",
    "audio_sample_rate",
    "onset_score",
    "onset_threshold",
    "onset_local_peak_score",
    "onset_confidence_ratio",
    "onset_baseline_duration",
    "onset_confirmation_time_absolute",
    "onset_confirmation_score",
    "onset_confirmation_threshold",
    "onset_backtrack_seconds",
    "end_motion_window_seconds",
    "end_motion_score_median",
    "end_motion_score_max",
    "end_motion_score_final",
    "end_motion_fraction_above_onset_threshold",
    "quality_warning",
    "status",
    "rejection_reason",
    "source_recording",
    "source_label_index",
    "source_imu_file",
    "source_audio_file",
    "source_label_file",
    "source_session_file",
    "imu_start_index",
    "imu_actual_crop_start",
    "imu_actual_crop_end_exclusive",
    "imu_start_alignment_error_s",
    "audio_start_index",
    "audio_actual_crop_start",
    "audio_actual_crop_end_exclusive",
    "audio_start_alignment_error_s",
    "audio_channels",
    "audio_sample_width_bytes",
    "label_fully_covered_by_imu",
    "label_fully_covered_by_audio",
    "imu_values_preserved_as_source_text",
    "audio_pcm_bit_exact",
    "sync_method",
    "imu_output_file",
    "audio_output_file",
    "diagnostic_plot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _ordered_sessions(sessions: list[SessionInfo]) -> list[SessionInfo]:
    ranking = {name: index for index, name in enumerate(DATASET_ORDER)}
    return sorted(sessions, key=lambda session: (ranking.get(session.dataset, 10_000), session.dataset))


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _nearest_nonnegative_index(value: float) -> int:
    if value < -1e-9:
        return -1
    return int(math.floor(value + 0.5))


def _public_destination(recording: str, source_index: int) -> tuple[str, str]:
    match = re.fullmatch(r"normal_train_(\d{2})", recording)
    if match:
        return "normal_train", f"session_{match.group(1)}_sample_{source_index:03d}"
    if recording in {"normal_validation", "normal_test", "slight_anomaly", "fierce_anomaly"}:
        return recording, f"sample_{source_index:03d}"
    raise ValueError(f"Unsupported raw recording name: {recording}")


def _base_metadata_row(
    input_root: Path,
    session: SessionInfo,
    label: LabelInterval,
) -> dict[str, object]:
    split, sample_id = _public_destination(session.dataset, label.source_index)
    return {
        "dataset": split,
        "sample_id": sample_id,
        "label_type": LABEL_NAMES.get(label.label, label.label),
        "label_confidence": label.confidence,
        "label_comment": label.comment,
        "original_label_start": label.start,
        "original_label_end": label.end,
        "original_label_duration": label.duration,
        "detected_t0_absolute": "",
        "t0_relative_to_label_start": "",
        "crop_start_absolute": "",
        "crop_end_absolute": "",
        "pre_available": "",
        "post_available": "",
        "imu_sample_count": 0,
        "audio_sample_count": 0,
        "imu_sample_rate": session.imu_track.frequency or "",
        "audio_sample_rate": session.audio_track.frequency or "",
        "onset_score": "",
        "onset_threshold": "",
        "onset_local_peak_score": "",
        "onset_confidence_ratio": "",
        "onset_baseline_duration": "",
        "onset_confirmation_time_absolute": "",
        "onset_confirmation_score": "",
        "onset_confirmation_threshold": "",
        "onset_backtrack_seconds": "",
        "end_motion_window_seconds": "",
        "end_motion_score_median": "",
        "end_motion_score_max": "",
        "end_motion_score_final": "",
        "end_motion_fraction_above_onset_threshold": "",
        "quality_warning": "",
        "status": "ERROR",
        "rejection_reason": "Not processed",
        "source_recording": session.dataset,
        "source_label_index": label.source_index,
        "source_imu_file": _relative(session.imu_track.payload_path, input_root),
        "source_audio_file": _relative(session.audio_track.payload_path, input_root),
        "source_label_file": _relative(session.label_track.payload_path, input_root),
        "source_session_file": _relative(session.session_path, input_root),
        "imu_start_index": "",
        "imu_actual_crop_start": "",
        "imu_actual_crop_end_exclusive": "",
        "imu_start_alignment_error_s": "",
        "audio_start_index": "",
        "audio_actual_crop_start": "",
        "audio_actual_crop_end_exclusive": "",
        "audio_start_alignment_error_s": "",
        "audio_channels": "",
        "audio_sample_width_bytes": "",
        "label_fully_covered_by_imu": "",
        "label_fully_covered_by_audio": "",
        "imu_values_preserved_as_source_text": False,
        "audio_pcm_bit_exact": False,
        "sync_method": (
            "IMU CSV absolute timestamps; Audio .imsession offset + WAV sample rate; "
            "nearest audio sample; no interpolation/resampling"
        ),
        "imu_output_file": "",
        "audio_output_file": "",
        "diagnostic_plot": "",
    }


def _add_onset_metadata(row: dict[str, object], onset: OnsetResult, label: LabelInterval) -> None:
    row["onset_threshold"] = onset.threshold
    row["onset_baseline_duration"] = onset.baseline_duration
    row["onset_confirmation_threshold"] = onset.confirmation_threshold
    row["onset_confirmation_time_absolute"] = (
        onset.confirmation_time if onset.confirmation_time is not None else ""
    )
    row["onset_confirmation_score"] = (
        onset.confirmation_score if onset.confirmation_score is not None else ""
    )
    row["onset_backtrack_seconds"] = (
        onset.backtrack_seconds if onset.backtrack_seconds is not None else ""
    )
    if onset.candidate_time is None:
        return
    t0 = onset.candidate_time
    row["detected_t0_absolute"] = t0
    row["t0_relative_to_label_start"] = t0 - label.start
    row["crop_start_absolute"] = t0 - PRE_SECONDS
    row["crop_end_absolute"] = t0 + POST_SECONDS
    row["pre_available"] = t0 - label.start
    row["post_available"] = label.end - t0
    row["onset_score"] = onset.onset_score if onset.onset_score is not None else ""
    row["onset_local_peak_score"] = (
        onset.local_peak_score if onset.local_peak_score is not None else ""
    )
    row["onset_confidence_ratio"] = (
        onset.confidence_ratio if onset.confidence_ratio is not None else ""
    )


def _verify_written_imu(
    output_path: Path,
    imu: ImuRecording,
    start_index: int,
    count: int,
    onset_index: int,
) -> None:
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["relative_time", "ax", "ay", "az", "gx", "gy", "gz"]:
            raise ValueError(f"Unexpected written IMU header in {output_path}")
        rows = list(reader)
    if len(rows) != count:
        raise ValueError(f"Written IMU row count {len(rows)} != {count} in {output_path}")
    for offset, row in enumerate(rows):
        source_index = start_index + offset
        if tuple(row[1:]) != imu.raw_axis_fields[source_index]:
            raise ValueError(f"Raw IMU field mismatch at output row {offset + 2}: {output_path}")
        expected_relative = imu.timestamps[source_index] - imu.timestamps[onset_index]
        if abs(float(row[0]) - expected_relative) > 5e-10:
            raise ValueError(f"Relative IMU time mismatch in {output_path}")


def _process_sample(
    input_root: Path,
    output_root: Path,
    session: SessionInfo,
    label: LabelInterval,
    imu: ImuRecording,
    audio: AudioRecording,
) -> tuple[dict[str, object], OnsetResult, np.ndarray | None, list[Path]]:
    row = _base_metadata_row(input_root, session, label)
    generated: list[Path] = []
    split = str(row["dataset"])
    sample_id = str(row["sample_id"])
    dataset_output = ensure_safe_output_path(output_root, output_root / split)
    imu_output = ensure_safe_output_path(
        output_root, dataset_output / f"{sample_id}_imu.csv"
    )
    audio_output = ensure_safe_output_path(
        output_root, dataset_output / f"{sample_id}_audio.wav"
    )
    diagnostic = ensure_safe_output_path(
        output_root,
        output_root
        / "diagnostics"
        / "samples"
        / split
        / f"{sample_id}_diagnostic.png",
    )
    # Exact generated names are safe to replace. This also removes stale files
    # from an interrupted run without touching unrelated output files.
    for stale in (imu_output, audio_output, diagnostic):
        if stale.is_file() or stale.is_symlink():
            stale.unlink()
    row["imu_sample_rate"] = imu.nominal_rate
    row["audio_sample_rate"] = audio.sample_rate
    row["audio_channels"] = audio.channels
    row["audio_sample_width_bytes"] = audio.sample_width

    imu_end_exclusive = float(imu.timestamps[-1] + imu.median_dt)
    audio_end_exclusive = float(
        session.audio_track.offset_seconds + audio.frame_count / audio.sample_rate
    )
    row["label_fully_covered_by_imu"] = bool(
        label.start >= imu.timestamps[0] - 1e-9 and label.end <= imu_end_exclusive + 1e-9
    )
    row["label_fully_covered_by_audio"] = bool(
        label.start >= session.audio_track.offset_seconds - 1e-9
        and label.end <= audio_end_exclusive + 1e-9
    )

    onset = detect_door_onset(
        imu.timestamps,
        imu.values,
        label.start,
        label.end,
        imu.nominal_rate,
    )
    _add_onset_metadata(row, onset, label)

    status = "OK"
    reason = ""
    score_trace: np.ndarray | None = None
    if onset.uncertain or onset.candidate_time is None or onset.candidate_global_index is None:
        status = "ONSET_UNCERTAIN"
        reason = onset.reason or "No reliable onset candidate"
    else:
        t0 = onset.candidate_time
        pre_available = t0 - label.start
        post_available = label.end - t0
        shortages: list[str] = []
        if pre_available < PRE_SECONDS - 1e-9:
            shortages.append(
                f"pre_available={pre_available:.9f}s < required {PRE_SECONDS:.1f}s"
            )
        if post_available < POST_SECONDS - 1e-9:
            shortages.append(
                f"post_available={post_available:.9f}s < required {POST_SECONDS:.1f}s"
            )
        if shortages:
            status = "REJECTED_INSUFFICIENT_LABEL"
            reason = "; ".join(shortages)

    if status == "OK":
        assert onset.candidate_time is not None
        assert onset.candidate_global_index is not None
        t0 = onset.candidate_time
        crop_start = t0 - PRE_SECONDS
        crop_end = t0 + POST_SECONDS
        imu_count = int(round(WINDOW_SECONDS * imu.nominal_rate))
        pre_count = int(round(PRE_SECONDS * imu.nominal_rate))
        audio_count = int(round(WINDOW_SECONDS * audio.sample_rate))
        if abs(imu_count / imu.nominal_rate - WINDOW_SECONDS) > 1e-9:
            status = "ERROR"
            reason = "IMU rate cannot represent the fixed 7.2 s window without resampling"
        elif not 719 <= imu_count <= 721:
            status = "ERROR"
            reason = f"IMU fixed-window shape {imu_count} is outside required 720±1"
        elif audio_count != 115200:
            status = "ERROR"
            reason = f"Audio fixed-window shape {audio_count} is not required 115200"
        else:
            imu_start = onset.candidate_global_index - pre_count
            imu_stop = imu_start + imu_count
            audio_position = (
                crop_start - session.audio_track.offset_seconds
            ) * audio.sample_rate
            audio_start = _nearest_nonnegative_index(audio_position)
            audio_stop = audio_start + audio_count
            imu_actual_start = (
                float(imu.timestamps[imu_start]) if 0 <= imu_start < imu.timestamps.size else float("nan")
            )
            imu_actual_end = (
                float(imu.timestamps[imu_stop - 1] + imu.median_dt)
                if 0 < imu_stop <= imu.timestamps.size
                else float("nan")
            )
            audio_actual_start = (
                session.audio_track.offset_seconds + audio_start / audio.sample_rate
                if audio_start >= 0
                else float("nan")
            )
            audio_actual_end = audio_actual_start + audio_count / audio.sample_rate

            row.update(
                {
                    "imu_start_index": imu_start,
                    "imu_actual_crop_start": imu_actual_start,
                    "imu_actual_crop_end_exclusive": imu_actual_end,
                    "imu_start_alignment_error_s": imu_actual_start - crop_start,
                    "audio_start_index": audio_start,
                    "audio_actual_crop_start": audio_actual_start,
                    "audio_actual_crop_end_exclusive": audio_actual_end,
                    "audio_start_alignment_error_s": audio_actual_start - crop_start,
                }
            )

            coverage_errors: list[str] = []
            if imu_start < 0 or imu_stop > imu.timestamps.size:
                coverage_errors.append("IMU stream does not cover requested crop")
            if audio_start < 0 or audio_stop > audio.frame_count:
                coverage_errors.append("Audio stream does not cover requested crop")
            if math.isfinite(imu_actual_start) and imu_actual_start < label.start - 1e-9:
                coverage_errors.append("Selected IMU begins before label hard boundary")
            if math.isfinite(imu_actual_end) and imu_actual_end > label.end + 1e-9:
                coverage_errors.append("Selected IMU ends after label hard boundary")
            if math.isfinite(audio_actual_start) and audio_actual_start < label.start - 1e-9:
                coverage_errors.append("Selected Audio begins before label hard boundary")
            if math.isfinite(audio_actual_end) and audio_actual_end > label.end + 1e-9:
                coverage_errors.append("Selected Audio ends after label hard boundary")
            if math.isfinite(imu_actual_start) and abs(imu_actual_start - crop_start) > 0.5 / imu.nominal_rate + 1e-9:
                coverage_errors.append("IMU nearest timestamp alignment error exceeds half a sample")
            if math.isfinite(audio_actual_start) and abs(audio_actual_start - crop_start) > 0.5 / audio.sample_rate + 1e-9:
                coverage_errors.append("Audio nearest-sample alignment error exceeds half a sample")
            if coverage_errors:
                status = "ERROR"
                reason = "; ".join(coverage_errors)
            else:
                write_imu_crop(
                    imu_output,
                    imu.raw_axis_fields,
                    imu.timestamps,
                    imu_start,
                    imu_count,
                    onset.candidate_global_index,
                )
                _verify_written_imu(
                    imu_output,
                    imu,
                    imu_start,
                    imu_count,
                    onset.candidate_global_index,
                )
                write_audio_crop(audio_output, audio, audio_start, audio_count)
                verify_written_audio(
                    audio_output,
                    audio.samples[audio_start:audio_stop],
                    audio.sample_rate,
                )
                row.update(
                    {
                        "imu_sample_count": imu_count,
                        "audio_sample_count": audio_count,
                        "imu_values_preserved_as_source_text": True,
                        "audio_pcm_bit_exact": True,
                        "imu_output_file": _relative(imu_output, output_root),
                        "audio_output_file": _relative(audio_output, output_root),
                    }
                )
                generated.extend((imu_output, audio_output))

                local_start = imu_start - int(onset.label_indices[0])
                local_stop = local_start + imu_count
                if local_start < 0 or local_stop > onset.motion_score.size:
                    raise ValueError("Diagnostic motion score does not cover accepted crop")
                score_trace = onset.motion_score[local_start:local_stop].copy()
                tail_seconds = 0.25
                tail_count = max(1, int(round(tail_seconds * imu.nominal_rate)))
                tail = score_trace[-tail_count:]
                fraction_above = float(np.mean(tail > onset.threshold))
                final_score = float(tail[-1])
                row.update(
                    {
                        "end_motion_window_seconds": tail_seconds,
                        "end_motion_score_median": float(np.median(tail)),
                        "end_motion_score_max": float(np.max(tail)),
                        "end_motion_score_final": final_score,
                        "end_motion_fraction_above_onset_threshold": fraction_above,
                    }
                )
                if fraction_above >= 0.20 or final_score > onset.threshold:
                    row["quality_warning"] = "POSSIBLE_ACTIVITY_AT_CROP_END"

    row["status"] = status
    row["rejection_reason"] = reason
    plot_reason = reason
    if row["quality_warning"]:
        plot_reason = (
            f"{plot_reason}; " if plot_reason else ""
        ) + str(row["quality_warning"])
    plot_sample_diagnostic(
        diagnostic,
        session.dataset,
        label,
        imu,
        audio,
        session.audio_track.offset_seconds,
        onset,
        status,
        plot_reason,
        PRE_SECONDS,
        POST_SECONDS,
    )
    row["diagnostic_plot"] = _relative(diagnostic, output_root)
    generated.append(diagnostic)
    return row, onset, score_trace, generated


def _write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_integrity_csv(
    path: Path,
    before: dict[str, dict[str, int | str]],
    after: dict[str, dict[str, int | str]],
) -> bool:
    all_names = sorted(set(before) | set(after))
    unchanged_all = True
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "relative_path",
            "size_before",
            "size_after",
            "mtime_ns_before",
            "mtime_ns_after",
            "sha256_before",
            "sha256_after",
            "unchanged",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in all_names:
            old = before.get(name, {})
            new = after.get(name, {})
            unchanged = bool(old and new and old == new)
            unchanged_all = unchanged_all and unchanged
            writer.writerow(
                {
                    "relative_path": name,
                    "size_before": old.get("size", ""),
                    "size_after": new.get("size", ""),
                    "mtime_ns_before": old.get("mtime_ns", ""),
                    "mtime_ns_after": new.get("mtime_ns", ""),
                    "sha256_before": old.get("sha256", ""),
                    "sha256_after": new.get("sha256", ""),
                    "unchanged": unchanged,
                }
            )
    return unchanged_all


def _sample_name(row: dict[str, object]) -> str:
    return f"{row['dataset']}/{row['sample_id']}"


def _fmt_list(rows: list[dict[str, object]]) -> str:
    return ", ".join(_sample_name(row) for row in rows) if rows else "None"


def _status_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(str(row["status"]) for row in rows)
    return {
        "total": len(rows),
        "ok": counts["OK"],
        "rejected": counts["REJECTED_INSUFFICIENT_LABEL"],
        "uncertain": counts["ONSET_UNCERTAIN"],
        "error": counts["ERROR"],
        "quality_warning": sum(bool(row.get("quality_warning")) for row in rows),
    }


def _onset_statistics(rows: list[dict[str, object]]) -> dict[str, float] | None:
    values = [
        float(row["t0_relative_to_label_start"])
        for row in rows
        if row["t0_relative_to_label_start"] != ""
        and row["status"] != "ONSET_UNCERTAIN"
    ]
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _build_report(
    input_root: Path,
    output_root: Path,
    rows: list[dict[str, object]],
    inspection: dict[str, object],
    integrity_unchanged: bool,
) -> str:
    summary = _status_summary(rows)
    pre_short = [
        row for row in rows if row["pre_available"] != "" and float(row["pre_available"]) < PRE_SECONDS
    ]
    post_short = [
        row for row in rows if row["post_available"] != "" and float(row["post_available"]) < POST_SECONDS
    ]
    uncertain = [row for row in rows if row["status"] == "ONSET_UNCERTAIN"]
    errors = [row for row in rows if row["status"] == "ERROR"]
    quality_warnings = [row for row in rows if row.get("quality_warning")]
    onset_stats = _onset_statistics(rows)

    lines = [
        "Audio + 6-axis IMU fixed-window crop report",
        "=" * 52,
        f"Original database (READ ONLY): {input_root}",
        f"Cleaned database: {output_root}",
        f"Crop window: [-{PRE_SECONDS:.1f}, +{POST_SECONDS:.1f}) s = {WINDOW_SECONDS:.1f} s",
        "No filtering, denoising, normalization, interpolation, padding, resampling, time warping, or sample-value editing was performed.",
        "",
        "Overall status",
        "--------------",
        f"Total human-labelled samples: {summary['total']}",
        f"OK: {summary['ok']}",
        f"REJECTED_INSUFFICIENT_LABEL: {summary['rejected']}",
        f"ONSET_UNCERTAIN: {summary['uncertain']}",
        f"ERROR: {summary['error']}",
        f"Successful crops with endpoint quality warning: {summary['quality_warning']}",
        "",
        "Per-dataset status",
        "------------------",
        f"{'Dataset':20s} {'total':>5s} {'OK':>5s} {'rejected':>9s} {'uncertain':>10s} {'error':>6s} {'warning':>8s}",
    ]
    ranking = {name: index for index, name in enumerate(DATASET_ORDER)}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    for dataset in sorted(grouped, key=lambda name: (ranking.get(name, 10_000), name)):
        item = _status_summary(grouped[dataset])
        lines.append(
            f"{dataset:20s} {item['total']:5d} {item['ok']:5d} {item['rejected']:9d} "
            f"{item['uncertain']:10d} {item['error']:6d} {item['quality_warning']:8d}"
        )

    lines.extend(("", "Door-onset timing statistics", "----------------------------"))
    if onset_stats:
        lines.extend(
            (
                "t0_relative_to_label_start over confident detections (seconds):",
                f"min={onset_stats['min']:.6f}",
                f"mean={onset_stats['mean']:.6f}",
                f"median={onset_stats['median']:.6f}",
                f"95th percentile={onset_stats['p95']:.6f}",
                f"max={onset_stats['max']:.6f}",
            )
        )
    else:
        lines.append("No confident onset detections.")
    lines.extend(
        (
            "Detector: initial-label robust per-axis noise normalization + causal 100 ms RMS;",
            "a persistent high-threshold crossing confirms motion, then adaptive hysteresis backtracks",
            "at most 1.25 s to the first departure connected across pauses no longer than 0.35 s.",
            "Only IMU samples inside the human label are inspected.",
        )
    )

    lines.extend(
        (
            "",
            "Samples with pre_available < 2.5 s",
            "----------------------------------",
            _fmt_list(pre_short),
            "",
            "Samples with post_available < 4.7 s",
            "-----------------------------------",
            _fmt_list(post_short),
            "",
            "ONSET_UNCERTAIN samples",
            "-----------------------",
        )
    )
    if uncertain:
        for row in uncertain:
            lines.append(
                f"{_sample_name(row)} candidate={row['detected_t0_absolute']} "
                f"score={row['onset_score']} threshold={row['onset_threshold']} "
                f"reason={row['rejection_reason']}"
            )
    else:
        lines.append("None")
    lines.extend(
        (
            "",
            "POSSIBLE_ACTIVITY_AT_CROP_END quality warnings",
            "----------------------------------------------",
            "These crops remain successful and the fixed +4.7 s boundary is unchanged.",
            "Flag rule: within the final 0.25 s, >=20% of score samples exceed the adaptive onset threshold,",
            "or the final score itself exceeds that threshold. Inspect the diagnostic to distinguish action tail from early reset.",
        )
    )
    if quality_warnings:
        for row in quality_warnings:
            lines.append(
                f"{_sample_name(row)}: median={float(row['end_motion_score_median']):.3f}, "
                f"max={float(row['end_motion_score_max']):.3f}, "
                f"final={float(row['end_motion_score_final']):.3f}, "
                f"fraction_above={float(row['end_motion_fraction_above_onset_threshold']):.3f}, "
                f"threshold={float(row['onset_threshold']):.3f}"
            )
    else:
        lines.append("None")
    lines.extend(("", "ERROR samples", "-------------"))
    if errors:
        for row in errors:
            lines.append(f"{_sample_name(row)}: {row['rejection_reason']}")
    else:
        lines.append("None")

    observed = {
        "normal_train/session_03_sample_015",
        "normal_train/session_05_sample_016",
        "normal_validation/sample_017",
    }
    actual = {_sample_name(row) for row in pre_short}
    lines.extend(
        (
            "",
            "Expected boundary rejections",
            "----------------------------",
            f"Recorded boundary cases: {', '.join(sorted(observed))}",
            f"Automatically detected: {', '.join(sorted(actual)) if actual else 'None'}",
            f"Exact match: {actual == observed}",
            "These records are checked after processing and are never used as a rejection list.",
            "",
            "Shape sanity checks",
            "-------------------",
        )
    )
    ok_rows = [row for row in rows if row["status"] == "OK"]
    imu_counts = Counter(int(row["imu_sample_count"]) for row in ok_rows)
    audio_counts = Counter(int(row["audio_sample_count"]) for row in ok_rows)
    lines.extend(
        (
            f"OK IMU sample-count distribution: {dict(sorted(imu_counts.items()))}",
            f"OK Audio sample-count distribution: {dict(sorted(audio_counts.items()))}",
            "Expected: IMU 720 raw rows; Audio 115200 raw PCM frames.",
            "IMU CSV axis fields were compared text-for-text with the source slice.",
            "Written WAV PCM frames were reread and compared bit-for-bit with the source slice.",
            "",
            "Time alignment",
            "--------------",
            "IMU: source CSV timestamps are already on the session timeline; the .imsession IMU offset is not added again.",
            "Audio: index = nearest((crop_start - per-session audio_offset) * actual_WAV_sample_rate).",
            "Nearest-sample alignment error is recorded per sample; no interpolation or resampling is used.",
            "The session files contain start offsets and nominal rates but no clock-drift model or sync pulses.",
            "",
            "Source inspection and metadata limitations",
            "------------------------------------------",
            f"Datasets: {inspection['dataset_count']}; source files: {inspection['file_count']}; labels: {inspection['total_labelled_samples']}",
            "Actual WAV: mono, 16 kHz, PCM16. Actual IMU: six axes, timestamped at 100 Hz.",
            "Accel ±8 g, gyro ±500 dps, and audio gain 5 were provided by the user but are absent from .imsession metadata.",
            "Observed quantization is consistent with the stated IMU ranges; this is supporting evidence, not explicit metadata.",
        )
    )
    warnings = inspection.get("global_warnings", [])
    if warnings:
        lines.append("Original-stream coverage warnings (diagnostics show the missing label tails):")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("Original-stream coverage warnings: None")
    lines.extend(
        (
            "",
            "Read-only integrity",
            "-------------------",
            f"All source size, mtime_ns, and SHA-256 values identical before/after run: {integrity_unchanged}",
            (
                "Original database was not modified."
                if integrity_unchanged
                else "WARNING: source integrity changed during the run; do not trust this run as read-only."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _build_readme(
    input_root: Path,
    output_root: Path,
    rows: list[dict[str, object]],
) -> str:
    summary = _status_summary(rows)
    rejected = [row for row in rows if row["status"] == "REJECTED_INSUFFICIENT_LABEL"]
    uncertain = [row for row in rows if row["status"] == "ONSET_UNCERTAIN"]
    errors = [row for row in rows if row["status"] == "ERROR"]
    warnings = [row for row in rows if row.get("quality_warning")]
    return f"""# Cleaned and Aligned Dataset

This directory was generated from the raw synchronized microphone and six-axis
IMU recordings. The source directory is treated as read-only and is hashed
before and after processing.

## Fixed data contract

- Window: `[-2.5, +4.7)` seconds around the detected door-motion onset.
- Audio: mono 16 kHz PCM16, 115,200 contiguous samples.
- IMU: 100 Hz, 720 contiguous raw rows, six axes.
- Synchronization: source timestamps and session offsets; no interpolation,
  resampling, padding, time warping, filtering, or sample-value editing.
- IMU output columns: `relative_time,ax,ay,az,gx,gy,gz`.
- Audio crops are verified bit-for-bit against the source segments.
- IMU axis fields are verified text-for-text against the source rows.

The onset detector inspects only the human-labelled interval. It estimates
per-axis difference noise from the initial quiet segment, confirms persistent
motion on a causal 100 ms normalized moving-RMS score, and backtracks through
short pauses to the first connected departure from the quiet baseline. The
score is used only for alignment and diagnostics; the saved IMU values remain
raw.

## Result

- Human-labelled intervals: {summary['total']}
- Accepted crops: {summary['ok']}
- Insufficient-label rejections: {summary['rejected']} ({_fmt_list(rejected)})
- Uncertain-onset records: {summary['uncertain']} ({_fmt_list(uncertain)})
- Processing errors: {summary['error']} ({_fmt_list(errors)})
- Accepted crops with an endpoint activity warning: {summary['quality_warning']} ({_fmt_list(warnings)})

An endpoint warning does not change the status, window, or data. It indicates
that the final 0.25 seconds still contain elevated motion-score activity and
should be reviewed with the diagnostic plot.

## Reproduce

From the repository root:

```powershell
python -m tools.clean_and_align_dataset --input dataset/raw --output training/outputs/processed_dataset
```

This stage does not create Mel features, normalize signals, train a model, fit
clusters, calculate anomaly scores, or perform deployment conversion.

Input: `{input_root}`
Output: `{output_root}`
"""

def main() -> int:
    args = parse_args()
    input_root, output_root = ensure_safe_roots(args.input, args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = ensure_safe_output_path(output_root, output_root / "metadata")
    diagnostics_samples = ensure_safe_output_path(
        output_root, output_root / "diagnostics" / "samples"
    )
    diagnostics_overview = ensure_safe_output_path(
        output_root, output_root / "diagnostics" / "overview"
    )
    for directory in (metadata_dir, diagnostics_samples, diagnostics_overview):
        directory.mkdir(parents=True, exist_ok=True)
        ensure_safe_output_path(output_root, directory)

    manifest_path = ensure_safe_output_path(
        output_root, metadata_dir / "generated_files_manifest.txt"
    )
    removed = remove_previous_generated_files(output_root, manifest_path)
    if removed:
        print(f"Safely removed {removed} files listed by the previous generated manifest.", flush=True)

    print("Hashing source files before processing...", flush=True)
    source_before = hash_source_tree(input_root)
    print("Inspecting actual database/session structure...", flush=True)
    inspection = inspect_database(input_root)
    inspection_path = ensure_safe_output_path(
        output_root, metadata_dir / "database_inspection.json"
    )
    write_json(inspection_path, inspection)
    generated_paths: list[Path] = [inspection_path]

    sessions = _ordered_sessions(discover_sessions(input_root))
    for session in sessions:
        public_split, _ = _public_destination(session.dataset, 1)
        dataset_directory = ensure_safe_output_path(
            output_root, output_root / public_split
        )
        diagnostic_directory = ensure_safe_output_path(
            output_root, diagnostics_samples / public_split
        )
        dataset_directory.mkdir(parents=True, exist_ok=True)
        diagnostic_directory.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, object]] = []
    overview_traces: dict[str, list[np.ndarray]] = defaultdict(list)
    overview_names: dict[str, list[str]] = defaultdict(list)
    overview_relative_times: np.ndarray | None = None
    error_log_lines: list[str] = []

    for session in sessions:
        labels = read_labels(session.label_track.payload_path)
        try:
            imu = read_imu(session.imu_track.payload_path, session.imu_track.frequency)
            audio = read_audio(session.audio_track.payload_path)
            if session.audio_track.frequency is not None and abs(audio.sample_rate - session.audio_track.frequency) > 1e-9:
                raise ValueError(
                    f"Audio metadata rate {session.audio_track.frequency} != WAV rate {audio.sample_rate}"
                )
            if abs(imu.nominal_rate - 100.0) > 1e-9 or abs(1.0 / imu.median_dt - 100.0) > 1e-6:
                raise ValueError(
                    f"Expected actual IMU rate 100 Hz, got metadata={imu.nominal_rate}, "
                    f"timestamps={1.0 / imu.median_dt}"
                )
            if audio.sample_rate != 16000:
                raise ValueError(f"Expected actual Audio rate 16000 Hz, got {audio.sample_rate}")
            if abs(imu.timestamps[0] - session.imu_track.offset_seconds) > 1e-8:
                raise ValueError(
                    "IMU first CSV timestamp does not match session offset; timeline semantics are ambiguous"
                )
        except Exception as exc:
            detail = "".join(traceback.format_exception(exc))
            error_log_lines.append(f"[{session.dataset}]\n{detail}")
            for label in labels:
                row = _base_metadata_row(input_root, session, label)
                reason = f"Dataset stream read/validation error: {exc}"
                row["rejection_reason"] = reason
                split, sample_id = _public_destination(session.dataset, label.source_index)
                diagnostic = ensure_safe_output_path(
                    output_root,
                    diagnostics_samples
                    / split
                    / f"{sample_id}_diagnostic.png",
                )
                try:
                    plot_error_diagnostic(diagnostic, session.dataset, label, reason)
                    row["diagnostic_plot"] = _relative(diagnostic, output_root)
                    generated_paths.append(diagnostic)
                except Exception as diagnostic_exc:
                    error_log_lines.append(
                        f"Error placeholder failure for {diagnostic}: {diagnostic_exc}"
                    )
                metadata_rows.append(row)
            print(f"{session.dataset}: all {len(labels)} samples ERROR ({exc})", flush=True)
            continue

        dataset_counts: Counter[str] = Counter()
        for label in labels:
            try:
                row, onset, trace, files = _process_sample(
                    input_root, output_root, session, label, imu, audio
                )
            except Exception as exc:
                detail = "".join(traceback.format_exception(exc))
                error_log_lines.append(
                    f"[{session.dataset}/sample_{label.source_index:03d}]\n{detail}"
                )
                row = _base_metadata_row(input_root, session, label)
                row["rejection_reason"] = f"Processing error: {exc}"
                split, sample_id = _public_destination(session.dataset, label.source_index)
                for partial in (
                    output_root / split / f"{sample_id}_imu.csv",
                    output_root / split / f"{sample_id}_audio.wav",
                    diagnostics_samples
                    / split
                    / f"{sample_id}_diagnostic.png",
                ):
                    try:
                        safe_partial = ensure_safe_output_path(output_root, partial)
                        if safe_partial.is_file() or safe_partial.is_symlink():
                            safe_partial.unlink()
                    except Exception as cleanup_exc:
                        error_log_lines.append(
                            f"Cleanup failure for {partial}: {cleanup_exc}"
                        )
                try:
                    onset = detect_door_onset(
                        imu.timestamps,
                        imu.values,
                        label.start,
                        label.end,
                        imu.nominal_rate,
                    )
                    _add_onset_metadata(row, onset, label)
                except Exception as onset_exc:
                    error_log_lines.append(
                        f"Fallback onset failure for {session.dataset}/sample_{label.source_index:03d}: {onset_exc}"
                    )
                diagnostic = ensure_safe_output_path(
                    output_root,
                    diagnostics_samples
                    / split
                    / f"{sample_id}_diagnostic.png",
                )
                try:
                    plot_error_diagnostic(
                        diagnostic,
                        session.dataset,
                        label,
                        str(row["rejection_reason"]),
                    )
                    row["diagnostic_plot"] = _relative(diagnostic, output_root)
                    files = [diagnostic]
                except Exception as diagnostic_exc:
                    error_log_lines.append(
                        f"Error placeholder failure for {diagnostic}: {diagnostic_exc}"
                    )
                    files = []
                trace = None
            metadata_rows.append(row)
            generated_paths.extend(files)
            dataset_counts[str(row["status"])] += 1
            if trace is not None and row["status"] == "OK":
                if overview_relative_times is None:
                    pre_count = int(round(PRE_SECONDS * imu.nominal_rate))
                    total_count = int(round(WINDOW_SECONDS * imu.nominal_rate))
                    overview_relative_times = (
                        np.arange(total_count, dtype=np.float64) - pre_count
                    ) / imu.nominal_rate
                if public_split == "normal_train":
                    group = "normal_training"
                elif public_split == "normal_validation":
                    group = "validation"
                elif public_split == "normal_test":
                    group = "test"
                elif public_split == "fierce_anomaly":
                    group = "abnormal_fierce"
                elif public_split == "slight_anomaly":
                    group = "abnormal_slight"
                else:
                    group = public_split
                overview_traces[group].append(trace)
                overview_names[group].append(_sample_name(row))
        print(
            f"{session.dataset}: total={len(labels)}, OK={dataset_counts['OK']}, "
            f"rejected={dataset_counts['REJECTED_INSUFFICIENT_LABEL']}, "
            f"uncertain={dataset_counts['ONSET_UNCERTAIN']}, error={dataset_counts['ERROR']}",
            flush=True,
        )

    overview_titles = {
        "normal_training": "Normal Train",
        "validation": "Normal Validation",
        "test": "Normal Test",
        "abnormal_fierce": "Fierce Anomaly",
        "abnormal_slight": "Slight Anomaly",
    }
    if overview_relative_times is None:
        overview_relative_times = np.arange(720, dtype=np.float64) / 100.0 - PRE_SECONDS
    for group, title in overview_titles.items():
        overview_path = ensure_safe_output_path(
            output_root, diagnostics_overview / f"{group}_overview.png"
        )
        plot_overview(
            overview_path,
            title,
            overview_relative_times,
            overview_traces.get(group, []),
            overview_names.get(group, []),
        )
        generated_paths.append(overview_path)

    metadata_path = ensure_safe_output_path(
        output_root, metadata_dir / "cleaning_metadata.csv"
    )
    _write_metadata(metadata_path, metadata_rows)
    generated_paths.append(metadata_path)
    if error_log_lines:
        error_log_path = ensure_safe_output_path(output_root, metadata_dir / "errors.log")
        error_log_path.write_text("\n\n".join(error_log_lines), encoding="utf-8")
        generated_paths.append(error_log_path)

    print("Hashing source files after processing...", flush=True)
    source_after = hash_source_tree(input_root)
    integrity_path = ensure_safe_output_path(
        output_root, metadata_dir / "source_integrity_sha256.csv"
    )
    integrity_unchanged = _write_integrity_csv(
        integrity_path, source_before, source_after
    )
    generated_paths.append(integrity_path)

    report_path = ensure_safe_output_path(output_root, output_root / "crop_report.txt")
    report_text = _build_report(
        input_root, output_root, metadata_rows, inspection, integrity_unchanged
    )
    report_path.write_text(report_text, encoding="utf-8", newline="\n")
    generated_paths.append(report_path)

    readme_path = ensure_safe_output_path(output_root, output_root / "README.md")
    readme_path.write_text(
        _build_readme(input_root, output_root, metadata_rows),
        encoding="utf-8",
        newline="\n",
    )
    generated_paths.append(readme_path)
    write_generated_manifest(output_root, manifest_path, generated_paths)

    if not integrity_unchanged:
        raise RuntimeError(
            "Source integrity mismatch: the original database changed during processing; "
            f"see {integrity_path}"
        )

    summary = _status_summary(metadata_rows)
    rejected_rows = [
        row for row in metadata_rows if row["status"] == "REJECTED_INSUFFICIENT_LABEL"
    ]
    uncertain_rows = [row for row in metadata_rows if row["status"] == "ONSET_UNCERTAIN"]
    error_rows = [row for row in metadata_rows if row["status"] == "ERROR"]

    print("\n" + report_text, flush=True)
    print("FINAL REQUIRED SUMMARY", flush=True)
    print(f"1. Original database path: {input_root}", flush=True)
    print(f"2. Cleaned database path: {output_root}", flush=True)
    print(f"3. Total labelled samples: {summary['total']}", flush=True)
    print(f"4. Successfully cropped samples: {summary['ok']}", flush=True)
    print(f"5. Rejected samples: {summary['rejected']}", flush=True)
    print(f"6. REJECTED_INSUFFICIENT_LABEL sample list: {_fmt_list(rejected_rows)}", flush=True)
    print(f"7. ONSET_UNCERTAIN sample list: {_fmt_list(uncertain_rows)}", flush=True)
    print(f"8. Error sample list: {_fmt_list(error_rows)}", flush=True)
    print(f"9. crop_metadata.csv absolute path: {metadata_path}", flush=True)
    print(f"10. crop_report.txt absolute path: {report_path}", flush=True)
    print(f"11. diagnostics folder absolute path: {diagnostics_samples}", flush=True)
    print(f"12. overview folder absolute path: {diagnostics_overview}", flush=True)
    print("Original database was not modified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
