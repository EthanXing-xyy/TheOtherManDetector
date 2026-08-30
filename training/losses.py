"""Joint reconstruction objective and modality-only diagnostics."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def reconstruction_losses(reconstruction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    if reconstruction.shape != target.shape or reconstruction.ndim != 3:
        raise ValueError(
            f"Reconstruction/target must have identical Bx180x64 shapes, got "
            f"{tuple(reconstruction.shape)} and {tuple(target.shape)}"
        )
    if tuple(target.shape[1:]) != (180, 64):
        raise ValueError(f"Expected Bx180x64 targets, got {tuple(target.shape)}")
    # The sole optimized objective is the unweighted MSE over every time x 64 element.
    joint = F.mse_loss(reconstruction, target, reduction="mean")
    audio = F.mse_loss(reconstruction[..., :40], target[..., :40], reduction="mean")
    imu = F.mse_loss(reconstruction[..., 40:], target[..., 40:], reduction="mean")
    return {"joint": joint, "audio": audio, "imu": imu}


def frame_residuals(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if reconstruction.shape != target.shape or tuple(target.shape[1:]) != (180, 64):
        raise ValueError("frame_residuals requires matching Bx180x64 tensors")
    return torch.mean(torch.square(reconstruction - target), dim=-1)

