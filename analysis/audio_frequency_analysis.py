#!/usr/bin/env python
"""Reproduce the 40 ms spectral-energy analysis used to select 0-7000 Hz.

This is an offline design analysis, not runtime feature preprocessing. It reads
the cleaned WAV files without changing them and uses only Normal Train when
selecting the frequency range.

Author: Ethan_Xing
License: MIT
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_RATE = 16_000
SAMPLE_COUNT = 115_200
FRAME_SAMPLES = 640
FRAME_COUNT = 180
FFT_SIZE = 1_024
FREQUENCY_BIN_COUNT = 513
FULL_SCALE_POWER = 32768.0**2
SELECTED_MIN_HZ = 0.0
SELECTED_MAX_HZ = 7_000.0
BANDS_HZ = (
    (0.0, 50.0),
    (50.0, 100.0),
    (100.0, 200.0),
    (200.0, 500.0),
    (500.0, 1_000.0),
    (1_000.0, 2_000.0),
    (2_000.0, 3_000.0),
    (3_000.0, 4_000.0),
    (4_000.0, 5_000.0),
    (5_000.0, 6_000.0),
    (6_000.0, 7_000.0),
    (7_000.0, 8_000.0),
)
CANDIDATE_RANGES_HZ = (
    ("full Nyquist range", 0.0, 8_000.0),
    ("candidate 0-5000 Hz", 0.0, 5_000.0),
    ("candidate 0-6000 Hz", 0.0, 6_000.0),
    ("candidate 0-6250 Hz", 0.0, 6_250.0),
    ("selected 0-7000 Hz", 0.0, 7_000.0),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo_root / "dataset" / "processed")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=repo_root / "dataset" / "processed" / "metadata" / "cleaning_metadata.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "analysis" / "results")
    return parser.parse_args()


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        properties = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
            source.getnframes(),
            source.getcomptype(),
        )
        raw = source.readframes(source.getnframes())
    expected = (1, 2, SAMPLE_RATE, SAMPLE_COUNT, "NONE")
    if properties != expected:
        raise ValueError(f"Unexpected WAV properties in {path}: {properties}")
    return np.frombuffer(raw, dtype="<i2").copy()


def local_power_spectrum(pcm: np.ndarray, window: np.ndarray) -> np.ndarray:
    frames = pcm.reshape(FRAME_COUNT, FRAME_SAMPLES).astype(np.float64)
    transform = np.fft.rfft(frames * window[None, :], n=FFT_SIZE, axis=1)
    power = np.abs(transform) ** 2 / (FFT_SIZE * np.sum(window**2))
    power[:, 1:-1] *= 2.0
    if power.shape != (FRAME_COUNT, FREQUENCY_BIN_COUNT):
        raise AssertionError(f"Unexpected spectrum shape: {power.shape}")
    return power.astype(np.float32)


def power_to_dbfs(power: np.ndarray | float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(power), FULL_SCALE_POWER * 1e-20) / FULL_SCALE_POWER)


def range_loss(
    label: str,
    low_hz: float,
    high_hz: float,
    frequencies: np.ndarray,
    spectra: np.ndarray,
) -> dict[str, float | int | str]:
    flat = spectra.reshape(-1, FREQUENCY_BIN_COUNT).astype(np.float64)
    total = flat.sum(axis=1)
    removed = flat[:, (frequencies < low_hz) | (frequencies > high_hz)].sum(axis=1)
    ratio = np.divide(removed, total, out=np.zeros_like(removed), where=total > 0)
    per_sample_max = ratio.reshape(-1, FRAME_COUNT).max(axis=1)
    return {
        "range_label": label,
        "f_min_hz": low_hz,
        "f_max_hz": high_hz,
        "aggregate_lost_energy_ratio": float(removed.sum() / total.sum()),
        "mean_frame_lost_ratio": float(ratio.mean()),
        "median_frame_lost_ratio": float(np.median(ratio)),
        "p95_frame_lost_ratio": float(np.percentile(ratio, 95)),
        "p99_frame_lost_ratio": float(np.percentile(ratio, 99)),
        "max_frame_lost_ratio": float(ratio.max()),
        "frames_lost_ratio_gt_5pct": int(np.count_nonzero(ratio > 0.05)),
        "frames_lost_ratio_gt_25pct": int(np.count_nonzero(ratio > 0.25)),
        "samples_with_any_frame_lost_ratio_gt_5pct": int(np.count_nonzero(per_sample_max > 0.05)),
        "samples_with_any_frame_lost_ratio_gt_25pct": int(np.count_nonzero(per_sample_max > 0.25)),
        "median_sample_max_frame_lost_ratio": float(np.median(per_sample_max)),
        "p95_sample_max_frame_lost_ratio": float(np.percentile(per_sample_max, 95)),
    }


def save_band_summary(
    output_dir: Path, frequencies: np.ndarray, train_spectra: np.ndarray
) -> None:
    flat = train_spectra.reshape(-1, FREQUENCY_BIN_COUNT).astype(np.float64)
    total = flat.sum(axis=1)
    rows = []
    for low, high in BANDS_HZ:
        if high == SAMPLE_RATE / 2:
            mask = (frequencies >= low) & (frequencies <= high)
        else:
            mask = (frequencies >= low) & (frequencies < high)
        energy = flat[:, mask].sum(axis=1)
        ratio = np.divide(energy, total, out=np.zeros_like(energy), where=total > 0)
        rows.append(
            {
                "band": f"{low:g}-{high:g} Hz",
                "low_hz_inclusive": low,
                "high_hz_exclusive_except_nyquist": high,
                "fft_bin_count": int(mask.sum()),
                "aggregate_energy_ratio": float(energy.sum() / total.sum()),
                "mean_energy_ratio": float(ratio.mean()),
                "median_energy_ratio": float(np.median(ratio)),
                "p95_energy_ratio": float(np.percentile(ratio, 95)),
                "p99_energy_ratio": float(np.percentile(ratio, 99)),
                "max_energy_ratio": float(ratio.max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "band_energy_summary.csv", index=False)


def save_cumulative_summary(
    output_dir: Path, frequencies: np.ndarray, mean_power: np.ndarray
) -> np.ndarray:
    cumulative = np.cumsum(mean_power, dtype=np.float64)
    ratio = cumulative / cumulative[-1]
    pd.DataFrame(
        {
            "frequency_hz": frequencies,
            "mean_power": mean_power,
            "cumulative_energy": cumulative,
            "cumulative_energy_ratio": ratio,
        }
    ).to_csv(output_dir / "cumulative_energy.csv", index=False)
    rows = []
    for target in (0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999):
        index = min(int(np.searchsorted(ratio, target)), len(frequencies) - 1)
        rows.append(
            {
                "cumulative_energy_target": target,
                "first_fft_bin_at_or_above_target_hz": frequencies[index],
                "actual_cumulative_energy_ratio": ratio[index],
                "fft_bin_index": index,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "cumulative_energy_thresholds.csv", index=False)
    return ratio


def save_plots(
    output_dir: Path,
    frequencies: np.ndarray,
    train_spectra: np.ndarray,
    mean_power: np.ndarray,
    cumulative_ratio: np.ndarray,
) -> None:
    flat = train_spectra.reshape(-1, FREQUENCY_BIN_COUNT)
    median = np.median(flat, axis=0)
    p95 = np.percentile(flat, 95, axis=0)
    p99 = np.percentile(flat, 99, axis=0)

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for values, label in ((mean_power, "Mean"), (median, "Median"), (p95, "P95"), (p99, "P99")):
        axis.plot(frequencies, power_to_dbfs(values), label=label)
    axis.axvline(SELECTED_MAX_HZ, color="black", linestyle="--", label="Selected maximum")
    axis.set(xlabel="Frequency (Hz)", ylabel="Power (dBFS)", title="Normal Train 40 ms spectral statistics")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "overall_spectrum.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(frequencies, cumulative_ratio)
    axis.axvline(SELECTED_MAX_HZ, color="black", linestyle="--")
    axis.set(xlabel="Frequency (Hz)", ylabel="Cumulative energy ratio", title="Normal Train cumulative spectral energy")
    axis.grid(alpha=0.25)
    fig.savefig(output_dir / "cumulative_energy.png", dpi=170)
    plt.close(fig)

    high = frequencies >= 5_000
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(frequencies[high], power_to_dbfs(mean_power[high]), label="Mean")
    axis.plot(frequencies[high], power_to_dbfs(p99[high]), label="P99")
    axis.axvline(SELECTED_MAX_HZ, color="black", linestyle="--")
    axis.set(xlabel="Frequency (Hz)", ylabel="Power (dBFS)", title="High-frequency detail")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "high_frequency_detail.png", dpi=170)
    plt.close(fig)

    frame_mean = train_spectra.mean(axis=0)
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    image = axis.imshow(
        power_to_dbfs(frame_mean).T,
        origin="lower",
        aspect="auto",
        extent=(-2.5, 4.7, 0, SAMPLE_RATE / 2),
        cmap="magma",
    )
    axis.axhline(SELECTED_MAX_HZ, color="white", linestyle="--")
    axis.set(xlabel="Time relative to aligned onset (s)", ylabel="Frequency (Hz)", title="Mean Normal Train time-frequency energy")
    fig.colorbar(image, ax=axis, label="Power (dBFS)")
    fig.savefig(output_dir / "time_frequency_overview.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == data_root or output_dir.is_relative_to(data_root):
        raise ValueError("Analysis output must be outside the read-only dataset")
    metadata = pd.read_csv(args.metadata.resolve(), dtype=str, keep_default_na=False)
    accepted = metadata.loc[metadata["status"] == "OK"].copy()
    train = accepted.loc[accepted["dataset"] == "normal_train"].copy()
    if len(accepted) != 164 or len(train) != 123:
        raise ValueError("Dataset counts do not match the fixed analysis contract")
    output_dir.mkdir(parents=True, exist_ok=True)
    window_index = np.arange(FRAME_SAMPLES, dtype=np.float64)
    periodic_hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * window_index / FRAME_SAMPLES)
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    spectra = []
    for row in train.itertuples(index=False):
        path = (data_root / row.audio_output_file).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError("Metadata audio path escapes the dataset root")
        spectra.append(local_power_spectrum(read_pcm16(path), periodic_hann))
    train_spectra = np.stack(spectra)
    flat = train_spectra.reshape(-1, FREQUENCY_BIN_COUNT)
    if flat.shape != (22_140, FREQUENCY_BIN_COUNT):
        raise AssertionError(f"Unexpected analysis matrix: {flat.shape}")
    mean_power = flat.mean(axis=0, dtype=np.float64)
    save_band_summary(output_dir, frequencies, train_spectra)
    cumulative_ratio = save_cumulative_summary(output_dir, frequencies, mean_power)
    losses = pd.DataFrame(
        [range_loss(label, low, high, frequencies, train_spectra) for label, low, high in CANDIDATE_RANGES_HZ]
    )
    losses.to_csv(output_dir / "candidate_range_loss_summary.csv", index=False)
    save_plots(output_dir, frequencies, train_spectra, mean_power, cumulative_ratio)
    selected = losses.loc[losses["range_label"] == "selected 0-7000 Hz"].iloc[0]
    if not math.isclose(float(selected["aggregate_lost_energy_ratio"]), 0.00416774, rel_tol=0.01):
        raise AssertionError("Selected-band aggregate loss differs from the verified analysis")
    print(
        "Selected 0-7000 Hz: "
        f"{100.0 * float(selected['aggregate_lost_energy_ratio']):.6f}% aggregate energy removed."
    )


if __name__ == "__main__":
    main()
