"""Reproducible structural inspection of the source database."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .dataset_io import discover_sessions, read_audio, read_imu, read_labels


def inspect_database(input_root: Path) -> dict[str, object]:
    input_root = input_root.resolve()
    sessions = discover_sessions(input_root)
    extension_counts = Counter(
        path.suffix.lower() for path in input_root.rglob("*") if path.is_file()
    )
    dataset_entries: list[dict[str, object]] = []
    total_labels = 0
    total_imu_rows = 0
    total_audio_frames = 0
    global_warnings: list[str] = []

    for session in sessions:
        for track in (session.label_track, session.imu_track, session.audio_track):
            if not track.payload_path.is_relative_to(input_root):
                raise ValueError(f"Session payload escapes input root: {track.payload_path}")
            if not track.payload_path.is_file():
                raise FileNotFoundError(f"Missing session payload: {track.payload_path}")

        labels = read_labels(session.label_track.payload_path)
        imu = read_imu(session.imu_track.payload_path, session.imu_track.frequency)
        audio = read_audio(session.audio_track.payload_path)
        total_labels += len(labels)
        total_imu_rows += imu.timestamps.size
        total_audio_frames += audio.frame_count

        imu_end_exclusive = float(imu.timestamps[-1] + imu.median_dt)
        audio_end_exclusive = float(
            session.audio_track.offset_seconds + audio.frame_count / audio.sample_rate
        )
        coverage_warnings: list[str] = []
        for label in labels:
            if label.end > imu_end_exclusive + 1e-9:
                coverage_warnings.append(
                    f"sample_{label.source_index:03d} label end exceeds IMU end by "
                    f"{label.end - imu_end_exclusive:.9f} s"
                )
            if label.end > audio_end_exclusive + 1e-9:
                coverage_warnings.append(
                    f"sample_{label.source_index:03d} label end exceeds audio end by "
                    f"{label.end - audio_end_exclusive:.9f} s"
                )
        global_warnings.extend(f"{session.dataset}: {item}" for item in coverage_warnings)

        label_types = sorted({label.label for label in labels})
        files = sorted(
            path.relative_to(input_root).as_posix()
            for path in session.session_path.parent.iterdir()
            if path.is_file()
        )
        entry = {
            "dataset": session.dataset,
            "session_file": session.session_path.relative_to(input_root).as_posix(),
            "studio_version": session.studio_version,
            "timeline_id": session.timeline_id,
            "files": files,
            "label": {
                "file": session.label_track.payload_path.relative_to(input_root).as_posix(),
                "count": len(labels),
                "types": label_types,
                "first_start_s": labels[0].start if labels else None,
                "last_end_s": labels[-1].end if labels else None,
                "minimum_duration_s": min((label.duration for label in labels), default=None),
                "maximum_duration_s": max((label.duration for label in labels), default=None),
            },
            "imu": {
                "file": session.imu_track.payload_path.relative_to(input_root).as_posix(),
                "metadata_frequency_hz": session.imu_track.frequency,
                "metadata_offset_s": session.imu_track.offset_seconds,
                "timecoded": session.imu_track.timecoded,
                "shape_labels": list(session.imu_track.shape_labels),
                "observed_rows": int(imu.timestamps.size),
                "observed_first_timestamp_s": float(imu.timestamps[0]),
                "observed_last_timestamp_s": float(imu.timestamps[-1]),
                "observed_end_exclusive_s": imu_end_exclusive,
                "observed_median_dt_s": imu.median_dt,
                "observed_rate_hz": 1.0 / imu.median_dt,
                "max_dt_error_s": imu.max_dt_error,
                "timestamp_already_contains_session_offset": bool(
                    abs(imu.timestamps[0] - session.imu_track.offset_seconds) < 1e-8
                ),
                "column_order": list(imu.header),
                "all_finite": bool(np.isfinite(imu.values).all()),
            },
            "audio": {
                "file": session.audio_track.payload_path.relative_to(input_root).as_posix(),
                "metadata_frequency_hz": session.audio_track.frequency,
                "metadata_offset_s": session.audio_track.offset_seconds,
                "metadata_shape_labels": list(session.audio_track.shape_labels),
                "observed_channels": audio.channels,
                "observed_sample_rate_hz": audio.sample_rate,
                "observed_sample_width_bytes": audio.sample_width,
                "observed_pcm_frames": audio.frame_count,
                "observed_duration_s": audio.frame_count / audio.sample_rate,
                "observed_end_exclusive_s": audio_end_exclusive,
                "compression_type": audio.compression_type,
            },
            "coverage_warnings": coverage_warnings,
        }
        dataset_entries.append(entry)

    return {
        "input_database": str(input_root),
        "read_only_policy": True,
        "dataset_count": len(sessions),
        "file_count": sum(extension_counts.values()),
        "extension_counts": dict(sorted(extension_counts.items())),
        "total_labelled_samples": total_labels,
        "total_imu_rows": total_imu_rows,
        "total_audio_frames": total_audio_frames,
        "metadata_limitations": {
            "accelerometer_range": "absent from .imsession; user states ±8 g",
            "gyroscope_range": "absent from .imsession; user states ±500 dps",
            "audio_gain": "absent from .imsession; user states gain 5",
            "imu_mode": "not explicit in .imsession; six columns/shape show Accel + Gyro",
            "clock_drift": "no drift coefficient or sync pulses; alignment uses per-session start offsets and nominal rates",
        },
        "global_warnings": global_warnings,
        "datasets": dataset_entries,
    }
