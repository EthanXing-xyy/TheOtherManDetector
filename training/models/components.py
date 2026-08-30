"""Shared encoder branches and the training-only reconstruction decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_NAMES = [f"audio_f{i:02d}" for i in range(32)] + [
    f"imu_f{i:02d}" for i in range(16)
]


class AudioEncoder(nn.Module):
    """Map normalized 180 x 40 log-Mel input to 180 x 32 features."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(40, 32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.GELU()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or tuple(audio.shape[1:]) != (180, 40):
            raise ValueError(f"AudioEncoder expected Bx180x40, got {tuple(audio.shape)}")
        features = self.activation(self.conv1(audio.transpose(1, 2)))
        features = self.activation(self.conv2(features)).transpose(1, 2)
        if tuple(features.shape[1:]) != (180, 32):
            raise AssertionError(f"Audio encoder shape mismatch: {tuple(features.shape)}")
        return features


class IMUEncoder(nn.Module):
    """Map normalized 720 x 6 IMU input to the common 180 x 16 grid."""

    def __init__(self) -> None:
        super().__init__()
        self.block_conv = nn.Conv1d(6, 12, kernel_size=4, stride=4, padding=0)
        self.context_conv = nn.Conv1d(12, 16, kernel_size=3, stride=1, padding=1)
        self.activation = nn.GELU()

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        if imu.ndim != 3 or tuple(imu.shape[1:]) != (720, 6):
            raise ValueError(f"IMUEncoder expected Bx720x6, got {tuple(imu.shape)}")
        features = self.activation(self.block_conv(imu.transpose(1, 2)))
        if tuple(features.shape[1:]) != (12, 180):
            raise AssertionError(f"IMU block convolution shape mismatch: {tuple(features.shape)}")
        features = self.activation(self.context_conv(features)).transpose(1, 2)
        if tuple(features.shape[1:]) != (180, 16):
            raise AssertionError(f"IMU encoder shape mismatch: {tuple(features.shape)}")
        return features


class ReconstructionDecoder(nn.Module):
    """Decode a 45 x 32 bottleneck into the 180 x 64 training target."""

    def __init__(self, upsample_mode: str = "nearest") -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.conv_45_to_90 = nn.Conv1d(32, 48, kernel_size=3, stride=1, padding=1)
        self.conv_90_to_180 = nn.Conv1d(48, 64, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Conv1d(64, 64, kernel_size=1, stride=1)
        self.activation = nn.GELU()

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        if bottleneck.ndim != 3 or tuple(bottleneck.shape[1:]) != (45, 32):
            raise ValueError(
                f"ReconstructionDecoder expected Bx45x32, got {tuple(bottleneck.shape)}"
            )
        decoded = bottleneck.transpose(1, 2)
        decoded = F.interpolate(decoded, scale_factor=2.0, mode=self.upsample_mode)
        decoded = self.activation(self.conv_45_to_90(decoded))
        decoded = F.interpolate(decoded, scale_factor=2.0, mode=self.upsample_mode)
        decoded = self.activation(self.conv_90_to_180(decoded))
        # Keep the reconstruction output linear for the joint MSE objective.
        output = self.output_conv(decoded).transpose(1, 2)
        if tuple(output.shape[1:]) != (180, 64):
            raise AssertionError(f"Decoder output shape mismatch: {tuple(output.shape)}")
        return output


def parameter_counts(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "model_size_fp32_bytes": int(total * 4),
        "model_size_fp32_mib": float(total * 4 / (1024**2)),
    }
