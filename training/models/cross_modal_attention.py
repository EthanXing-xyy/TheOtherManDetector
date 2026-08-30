"""Vectorized bipartite cross-modal feature attention."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CrossModalAttention(nn.Module):
    """Attend only between the 32 audio and 16 IMU feature partitions."""

    def __init__(
        self,
        audio_feature_count: int = 32,
        imu_feature_count: int = 16,
        trajectory_length: int = 90,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.audio_feature_count = int(audio_feature_count)
        self.imu_feature_count = int(imu_feature_count)
        self.feature_count = self.audio_feature_count + self.imu_feature_count
        self.trajectory_length = int(trajectory_length)
        self.negative_slope = float(negative_slope)
        self.attention_vector = nn.Parameter(torch.empty(2 * self.trajectory_length))
        nn.init.xavier_uniform_(self.attention_vector.view(1, -1))

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (self.trajectory_length, self.feature_count)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                f"CrossModalAttention expected Bx{expected[0]}x{expected[1]}, "
                f"got {tuple(values.shape)}"
            )
        nodes = values.transpose(1, 2)
        audio_nodes = nodes[:, : self.audio_feature_count, :]
        imu_nodes = nodes[:, self.audio_feature_count :, :]
        receiver_weights, source_weights = self.attention_vector.chunk(2)

        audio_receiver = torch.einsum("bnd,d->bn", audio_nodes, receiver_weights)
        audio_source = torch.einsum("bnd,d->bn", audio_nodes, source_weights)
        imu_receiver = torch.einsum("bnd,d->bn", imu_nodes, receiver_weights)
        imu_source = torch.einsum("bnd,d->bn", imu_nodes, source_weights)

        audio_to_imu_scores = F.leaky_relu(
            audio_receiver.unsqueeze(-1) + imu_source.unsqueeze(-2),
            negative_slope=self.negative_slope,
        )
        audio_to_imu = torch.softmax(audio_to_imu_scores, dim=-1)
        audio_output = torch.sigmoid(torch.bmm(audio_to_imu, imu_nodes))

        imu_to_audio_scores = F.leaky_relu(
            imu_receiver.unsqueeze(-1) + audio_source.unsqueeze(-2),
            negative_slope=self.negative_slope,
        )
        imu_to_audio = torch.softmax(imu_to_audio_scores, dim=-1)
        imu_output = torch.sigmoid(torch.bmm(imu_to_audio, audio_nodes))

        output = torch.cat((audio_output, imu_output), dim=1).transpose(1, 2)
        if output.shape != values.shape:
            raise AssertionError(f"Cross-modal output shape mismatch: {tuple(output.shape)}")
        return output, {
            "audio_to_imu": audio_to_imu,
            "imu_to_audio": imu_to_audio,
        }
