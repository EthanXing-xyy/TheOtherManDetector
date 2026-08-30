"""Deterministic 40 ms block-aligned log-Mel extraction.

No sample-dependent amplitude reference or normalization is applied here.  PCM
is mapped to a consistent full-scale floating representation, and the only
normalization later applied is the Normal-Train global per-Mel-band z-score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import hann


def hz_to_slaney_mel(frequencies_hz: np.ndarray) -> np.ndarray:
    """Convert Hz to the Auditory Toolbox/librosa Slaney Mel convention."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mels = frequencies_hz / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = frequencies_hz >= min_log_hz
    mels[log_region] = min_log_mel + np.log(
        frequencies_hz[log_region] / min_log_hz
    ) / logstep
    return mels


def slaney_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """Convert Slaney Mels to Hz."""
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    frequencies = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    frequencies[log_region] = min_log_hz * np.exp(
        logstep * (mels[log_region] - min_log_mel)
    )
    return frequencies


def create_slaney_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Return a Slaney-scale, Slaney-area-normalized Mel filterbank.

    The implementation follows the triangular filter construction and
    `2 / (f[m+2] - f[m])` area normalization used by librosa/Slaney.
    Shape is ``(n_mels, n_fft // 2 + 1)``.
    """
    if not (0.0 <= f_min < f_max <= sample_rate / 2.0):
        raise ValueError(f"Invalid Mel range [{f_min}, {f_max}] for sr={sample_rate}")
    fft_frequencies = np.linspace(0.0, sample_rate / 2.0, 1 + n_fft // 2)
    mel_min, mel_max = hz_to_slaney_mel(np.array([f_min, f_max]))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    mel_frequencies = slaney_mel_to_hz(mel_points)

    ramps = mel_frequencies[:, None] - fft_frequencies[None, :]
    fdiff = np.diff(mel_frequencies)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_frequencies[2:] - mel_frequencies[:-2]))[:, None]
    weights = weights.astype(np.float32)
    if weights.shape != (n_mels, n_fft // 2 + 1):
        raise AssertionError(f"Mel filterbank shape mismatch: {weights.shape}")
    if not np.isfinite(weights).all() or np.any(weights.sum(axis=1) <= 0):
        raise ValueError("Mel filterbank contains invalid or empty filters")
    return weights


def pcm_to_float32(pcm: np.ndarray) -> np.ndarray:
    """Map integer PCM to [-1, 1) using its type's full-scale denominator."""
    if np.issubdtype(pcm.dtype, np.signedinteger):
        info = np.iinfo(pcm.dtype)
        scale = float(max(abs(info.min), info.max + 1))
        return pcm.astype(np.float32) / scale
    if np.issubdtype(pcm.dtype, np.unsignedinteger):
        info = np.iinfo(pcm.dtype)
        midpoint = (info.max + 1) / 2.0
        return (pcm.astype(np.float32) - midpoint) / midpoint
    if np.issubdtype(pcm.dtype, np.floating):
        return pcm.astype(np.float32, copy=False)
    raise TypeError(f"Unsupported WAV PCM dtype: {pcm.dtype}")


def load_wav_mono(path: str | Path, expected_rate: int, expected_samples: int) -> np.ndarray:
    sample_rate, pcm = wavfile.read(Path(path))
    if int(sample_rate) != int(expected_rate):
        raise ValueError(f"{path}: expected {expected_rate} Hz, got {sample_rate}")
    if pcm.ndim != 1:
        raise ValueError(f"{path}: expected mono WAV, got shape {pcm.shape}")
    if pcm.dtype != np.int16:
        raise ValueError(f"{path}: expected PCM16/int16, got {pcm.dtype}")
    if len(pcm) != expected_samples:
        raise ValueError(f"{path}: expected {expected_samples} samples, got {len(pcm)}")
    audio = pcm_to_float32(pcm)
    if not np.isfinite(audio).all():
        raise ValueError(f"{path}: non-finite PCM values")
    return audio


def extract_log_mel(audio: np.ndarray, config: dict[str, Any], filterbank: np.ndarray) -> np.ndarray:
    """Extract exactly 180 x 40 log-Mel energies from one 7.2 s waveform."""
    samples = int(config["samples"])
    block = int(config["samples_per_block"])
    frames = samples // block
    if audio.shape != (samples,):
        raise ValueError(f"Expected audio shape {(samples,)}, got {audio.shape}")
    if samples % block:
        raise ValueError("Audio sample count is not divisible by block size")
    blocks = audio.reshape(frames, block)
    window = hann(block, sym=not bool(config["hann_periodic"])).astype(np.float32)
    windowed = blocks * window[None, :]
    spectrum = np.fft.rfft(windowed, n=int(config["n_fft"]), axis=1)
    power = np.abs(spectrum) ** 2
    expected_bins = int(config["n_fft"]) // 2 + 1
    if power.shape != (frames, expected_bins):
        raise AssertionError(f"Power spectrum shape mismatch: {power.shape}")
    mel_energy = power @ filterbank.T
    log_mel = np.log(mel_energy + float(config["eps"])).astype(np.float32)
    expected = (frames, int(config["n_mels"]))
    if log_mel.shape != expected:
        raise AssertionError(f"Log-Mel shape mismatch: {log_mel.shape} != {expected}")
    if not np.isfinite(log_mel).all():
        raise ValueError("Non-finite log-Mel values")
    return log_mel


def load_and_extract(path: str | Path, config: dict[str, Any], filterbank: np.ndarray) -> np.ndarray:
    audio = load_wav_mono(path, int(config["sample_rate"]), int(config["samples"]))
    return extract_log_mel(audio, config, filterbank)
