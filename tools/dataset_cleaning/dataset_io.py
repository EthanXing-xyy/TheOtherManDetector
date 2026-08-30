"""Read-only parsers and bit-preserving crop writers for the multimodal dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import wave
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EXPECTED_IMU_COLUMNS = (
    "Time (seconds)",
    "Accel_X",
    "Accel_Y",
    "Accel_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z",
)


@dataclass(frozen=True)
class TrackInfo:
    name: str
    track_type: str
    payload_path: Path
    frequency: float | None
    offset_seconds: float
    timecoded: str | None
    shape_labels: tuple[str, ...]


@dataclass(frozen=True)
class SessionInfo:
    dataset: str
    session_path: Path
    studio_version: str
    timeline_id: str
    label_track: TrackInfo
    imu_track: TrackInfo
    audio_track: TrackInfo


@dataclass(frozen=True)
class LabelInterval:
    source_index: int
    start: float
    duration: float
    end: float
    label: str
    confidence: float
    comment: str


@dataclass
class ImuRecording:
    path: Path
    timestamps: np.ndarray
    values: np.ndarray
    raw_axis_fields: list[tuple[str, str, str, str, str, str]]
    header: tuple[str, ...]
    nominal_rate: float
    median_dt: float
    max_dt_error: float


@dataclass(frozen=True)
class AudioRecording:
    path: Path
    channels: int
    sample_rate: int
    sample_width: int
    frame_count: int
    compression_type: str
    samples: np.ndarray


def _offset_seconds(track: ET.Element) -> float:
    offset = track.find("Offset")
    if offset is None or offset.text is None:
        return 0.0
    unit = offset.attrib.get("unit", "seconds").strip().lower()
    value = float(offset.text)
    factors = {
        "seconds": 1.0,
        "second": 1.0,
        "s": 1.0,
        "milliseconds": 1e-3,
        "millisecond": 1e-3,
        "ms": 1e-3,
        "microseconds": 1e-6,
        "microsecond": 1e-6,
        "us": 1e-6,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported Offset unit {unit!r} in track metadata")
    return value * factors[unit]


def _parse_track(track: ET.Element, session_dir: Path) -> TrackInfo:
    payload_node = track.find("PayloadFile")
    if payload_node is None or not payload_node.text:
        raise ValueError("Session track has no PayloadFile")
    frequency_text = track.attrib.get("frequency")
    labels: list[str] = []
    for node in track.findall("./Shape/Axis/Label"):
        labels.append((node.text or "").strip())
    return TrackInfo(
        name=track.attrib.get("name", ""),
        track_type=track.attrib.get("type", ""),
        payload_path=(session_dir / payload_node.text.strip()).resolve(),
        frequency=float(frequency_text) if frequency_text is not None else None,
        offset_seconds=_offset_seconds(track),
        timecoded=track.attrib.get("timecoded"),
        shape_labels=tuple(labels),
    )


def parse_session(session_path: Path) -> SessionInfo:
    session_path = session_path.resolve()
    root = ET.parse(session_path).getroot()
    timeline = root.find("Timeline")
    if timeline is None:
        raise ValueError(f"No Timeline in {session_path}")
    tracks = [
        _parse_track(node, session_path.parent)
        for node in timeline.findall("./Tracks/Track")
    ]
    by_type: dict[str, list[TrackInfo]] = {}
    for track in tracks:
        by_type.setdefault(track.track_type, []).append(track)
    required = {"labelcsv": "label", "datacsv": "IMU", "audiowav": "audio"}
    selected: dict[str, TrackInfo] = {}
    for track_type, description in required.items():
        matches = by_type.get(track_type, [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {description} ({track_type}) track in "
                f"{session_path}, found {len(matches)}"
            )
        selected[track_type] = matches[0]
    return SessionInfo(
        dataset=session_path.parent.name,
        session_path=session_path,
        studio_version=root.attrib.get("version", ""),
        timeline_id=timeline.attrib.get("id", ""),
        label_track=selected["labelcsv"],
        imu_track=selected["datacsv"],
        audio_track=selected["audiowav"],
    )


def discover_sessions(input_root: Path) -> list[SessionInfo]:
    input_root = input_root.resolve()
    sessions: list[SessionInfo] = []
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        candidates = sorted(dataset_dir.glob("*.imsession"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one .imsession in {dataset_dir}, found {len(candidates)}"
            )
        sessions.append(parse_session(candidates[0]))
    if not sessions:
        raise ValueError(f"No sessions found under {input_root}")
    return sessions


def read_labels(path: Path) -> list[LabelInterval]:
    labels: list[LabelInterval] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Time(Seconds)",
            "Length(Seconds)",
            "Label(string)",
            "Confidence(double)",
            "Comment(string)",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected label header in {path}: {reader.fieldnames}")
        previous_end = -math.inf
        for index, row in enumerate(reader, start=1):
            start = float(row["Time(Seconds)"])
            duration = float(row["Length(Seconds)"])
            end = start + duration
            if not all(math.isfinite(value) for value in (start, duration, end)):
                raise ValueError(f"Non-finite label time at row {index} in {path}")
            if duration <= 0:
                raise ValueError(f"Non-positive label duration at row {index} in {path}")
            if start < previous_end - 1e-12:
                raise ValueError(f"Overlapping/out-of-order label at row {index} in {path}")
            previous_end = end
            labels.append(
                LabelInterval(
                    source_index=index,
                    start=start,
                    duration=duration,
                    end=end,
                    label=row["Label(string)"],
                    confidence=float(row["Confidence(double)"]),
                    comment=row["Comment(string)"],
                )
            )
    return labels


def read_imu(path: Path, declared_rate: float | None) -> ImuRecording:
    timestamps: list[float] = []
    values: list[tuple[float, float, float, float, float, float]] = []
    raw_fields: list[tuple[str, str, str, str, str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline().rstrip("\r\n")
        normalized = first_line[1:].lstrip() if first_line.startswith("#") else first_line
        header = tuple(next(csv.reader([normalized])))
        if header != EXPECTED_IMU_COLUMNS:
            raise ValueError(f"Unexpected IMU header in {path}: {header}")
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 7:
                raise ValueError(f"IMU row {row_number} in {path} has {len(row)} columns")
            numeric = tuple(float(field) for field in row)
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"Non-finite IMU value at row {row_number} in {path}")
            timestamps.append(numeric[0])
            values.append(numeric[1:])
            raw_fields.append(tuple(row[1:]))
    if len(timestamps) < 2:
        raise ValueError(f"Too few IMU rows in {path}")
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    differences = np.diff(timestamp_array)
    if np.any(differences <= 0):
        raise ValueError(f"IMU timestamps are not strictly increasing in {path}")
    median_dt = float(np.median(differences))
    nominal_rate = float(declared_rate) if declared_rate else 1.0 / median_dt
    expected_dt = 1.0 / nominal_rate
    max_dt_error = float(np.max(np.abs(differences - expected_dt)))
    tolerance = max(1e-9, expected_dt * 1e-6)
    if max_dt_error > tolerance:
        raise ValueError(
            f"IMU timestamp spacing in {path} is irregular: max error {max_dt_error:.9g}s"
        )
    return ImuRecording(
        path=path.resolve(),
        timestamps=timestamp_array,
        values=value_array,
        raw_axis_fields=raw_fields,
        header=header,
        nominal_rate=nominal_rate,
        median_dt=median_dt,
        max_dt_error=max_dt_error,
    )


def read_audio(path: Path) -> AudioRecording:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frame_count = source.getnframes()
        compression_type = source.getcomptype()
        raw = source.readframes(frame_count)
    if compression_type != "NONE":
        raise ValueError(f"Compressed WAV is unsupported: {path} ({compression_type})")
    if channels != 1:
        raise ValueError(f"Expected mono WAV, got {channels} channels in {path}")
    if sample_width != 2:
        raise ValueError(f"Expected PCM16 WAV, got {sample_width * 8}-bit in {path}")
    expected_bytes = frame_count * channels * sample_width
    if len(raw) != expected_bytes:
        raise ValueError(f"Truncated WAV frame data in {path}")
    samples = np.frombuffer(raw, dtype="<i2").copy()
    return AudioRecording(
        path=path.resolve(),
        channels=channels,
        sample_rate=sample_rate,
        sample_width=sample_width,
        frame_count=frame_count,
        compression_type=compression_type,
        samples=samples,
    )


def write_imu_crop(
    output_path: Path,
    raw_axis_fields: list[tuple[str, str, str, str, str, str]],
    timestamps: np.ndarray,
    start_index: int,
    count: int,
    onset_index: int,
) -> None:
    """Write original axis strings unchanged; only relative_time is newly computed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output_path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("relative_time", "ax", "ay", "az", "gx", "gy", "gz"))
            for source_index in range(start_index, start_index + count):
                relative_time = timestamps[source_index] - timestamps[onset_index]
                writer.writerow((f"{relative_time:.9f}", *raw_axis_fields[source_index]))
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_audio_crop(
    output_path: Path,
    recording: AudioRecording,
    start_index: int,
    count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment = recording.samples[start_index : start_index + count]
    if segment.size != count:
        raise ValueError(
            f"Audio slice has {segment.size} frames; expected {count} for {output_path}"
        )
    temporary = _temporary_sibling(output_path)
    try:
        with wave.open(str(temporary), "wb") as target:
            target.setnchannels(recording.channels)
            target.setsampwidth(recording.sample_width)
            target.setframerate(recording.sample_rate)
            target.setcomptype("NONE", "not compressed")
            target.writeframes(segment.astype("<i2", copy=False).tobytes())
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def verify_written_audio(
    output_path: Path,
    expected: np.ndarray,
    expected_rate: int,
) -> None:
    reread = read_audio(output_path)
    if reread.sample_rate != expected_rate or reread.frame_count != expected.size:
        raise ValueError(f"Written WAV header verification failed for {output_path}")
    if not np.array_equal(reread.samples, expected):
        raise ValueError(f"Written WAV samples are not bit-identical for {output_path}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_source_tree(input_root: Path) -> dict[str, dict[str, int | str]]:
    input_root = input_root.resolve()
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(item for item in input_root.rglob("*") if item.is_file()):
        stat = path.stat()
        relative = path.relative_to(input_root).as_posix()
        result[relative] = {
            "sha256": sha256_file(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def ensure_safe_roots(input_root: Path, output_root: Path) -> tuple[Path, Path]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input database does not exist: {input_root}")
    if (
        input_root == output_root
        or output_root.is_relative_to(input_root)
        or input_root.is_relative_to(output_root)
    ):
        raise ValueError(
            "Input and output roots must be disjoint (neither may contain the other); "
            "refusing unsafe paths"
        )
    return input_root, output_root


def ensure_safe_output_path(output_root: Path, candidate: Path) -> Path:
    """Reject output paths whose symlink-resolved destination escapes output_root."""
    output_root = output_root.resolve()
    candidate = candidate if candidate.is_absolute() else output_root / candidate
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(output_root):
        raise ValueError(f"Output path escapes output root (possibly via symlink): {candidate}")
    return candidate


def remove_previous_generated_files(output_root: Path, manifest_path: Path) -> int:
    """Remove only files recorded by this script, never an output directory wholesale."""
    if not manifest_path.exists():
        return 0
    removed = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        relative_paths = [line.strip() for line in handle if line.strip()]
    for relative_text in relative_paths:
        candidate = output_root / Path(relative_text)
        resolved_candidate = candidate.resolve(strict=False)
        if not resolved_candidate.is_relative_to(output_root):
            raise ValueError(f"Unsafe path in generated manifest: {relative_text}")
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink()
            removed += 1
    return removed


def write_generated_manifest(
    output_root: Path,
    manifest_path: Path,
    generated_paths: Iterable[Path],
) -> None:
    relatives: set[str] = set()
    for path in generated_paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(output_root):
            raise ValueError(f"Generated file escaped output root: {resolved}")
        if resolved == manifest_path.resolve():
            continue
        relatives.add(resolved.relative_to(output_root).as_posix())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(manifest_path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for relative in sorted(relatives):
                handle.write(relative + "\n")
        temporary.replace(manifest_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _temporary_sibling(output_path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    return Path(name)
