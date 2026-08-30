"""Multimodal reconstruction model used to learn the deployed representation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .components import AudioEncoder, FEATURE_NAMES, IMUEncoder, ReconstructionDecoder
from .cross_modal_attention import CrossModalAttention
from .temporal_attention import TemporalAttention


PHYSICAL_FRAME_COUNT = 180
GLOBAL_TOKEN_COUNT = 90
GLOBAL_TOKEN_MILLISECONDS = 80
POOL_KERNEL = 2
POOL_STRIDE = 2
FIRST_BOTTLENECK_TOKENS = 45
FINAL_BOTTLENECK_TOKENS = 23
DECODER_INPUT_TOKENS = 45
EMBEDDING_DIMENSION = 32


class TrainingModel(nn.Module):
    """Reconstruction network with a 32-channel temporal bottleneck.

    The modality encoders operate on the common 180-frame, 40 ms grid. A
    fixed average pool maps those local features to 90 global tokens before
    cross-modal attention, temporal attention, and recurrent modeling. The
    decoder is used only for the reconstruction objective.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config["model"]
        self.audio_branch = AudioEncoder()
        self.imu_branch = IMUEncoder()
        self.temporal_downsample = nn.AvgPool1d(
            kernel_size=POOL_KERNEL,
            stride=POOL_STRIDE,
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
        )
        self.feature_gat = CrossModalAttention(
            audio_feature_count=32,
            imu_feature_count=16,
            trajectory_length=GLOBAL_TOKEN_COUNT,
            negative_slope=float(model_config["gat_negative_slope"]),
        )
        self.time_gat = TemporalAttention(
            token_count=GLOBAL_TOKEN_COUNT,
            feature_dim=int(model_config["time_gat_node_dim"]),
            negative_slope=float(model_config["gat_negative_slope"]),
        )
        self.gru = nn.GRU(
            input_size=int(model_config["gru_input_size"]),
            hidden_size=int(model_config["gru_hidden_size"]),
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.bottleneck1 = nn.Conv1d(64, 48, kernel_size=3, stride=2, padding=1)
        self.bottleneck2 = nn.Conv1d(48, 32, kernel_size=3, stride=2, padding=1)
        self.activation = nn.GELU()
        self.decoder = ReconstructionDecoder(str(model_config["upsample_mode"]))
        self.feature_names = FEATURE_NAMES.copy()

    def _downsample_local_features(
        self, features: torch.Tensor, channels: int
    ) -> torch.Tensor:
        expected = (PHYSICAL_FRAME_COUNT, channels)
        if features.ndim != 3 or tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"Temporal downsampling expected Bx{expected[0]}x{expected[1]}, "
                f"got {tuple(features.shape)}"
            )
        pooled = self.temporal_downsample(features.transpose(1, 2)).transpose(1, 2)
        if tuple(pooled.shape[1:]) != (GLOBAL_TOKEN_COUNT, channels):
            raise AssertionError(f"Temporal downsampling shape mismatch: {tuple(pooled.shape)}")
        return pooled

    def encode(
        self, audio: torch.Tensor, imu: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        audio_local = self.audio_branch(audio)
        imu_local = self.imu_branch(imu)
        audio_features = self._downsample_local_features(audio_local, 32)
        imu_features = self._downsample_local_features(imu_local, 16)
        multimodal = torch.cat((audio_features, imu_features), dim=-1)
        if tuple(multimodal.shape[1:]) != (GLOBAL_TOKEN_COUNT, 48):
            raise AssertionError(f"Multimodal shape mismatch: {tuple(multimodal.shape)}")

        cross_features, cross_attention = self.feature_gat(multimodal)
        temporal_features, temporal_attention = self.time_gat(multimodal)
        recurrent_input = torch.cat(
            (multimodal, cross_features, temporal_features), dim=-1
        )
        if tuple(recurrent_input.shape[1:]) != (GLOBAL_TOKEN_COUNT, 144):
            raise AssertionError(f"Recurrent input shape mismatch: {tuple(recurrent_input.shape)}")

        sequence, hidden = self.gru(recurrent_input)
        if tuple(sequence.shape[1:]) != (GLOBAL_TOKEN_COUNT, 64):
            raise AssertionError(f"GRU output shape mismatch: {tuple(sequence.shape)}")
        first = self.activation(self.bottleneck1(sequence.transpose(1, 2)))
        if tuple(first.shape[1:]) != (48, FIRST_BOTTLENECK_TOKENS):
            raise AssertionError(f"First bottleneck shape mismatch: {tuple(first.shape)}")
        final_channels_first = self.activation(self.bottleneck2(first))
        if tuple(final_channels_first.shape[1:]) != (32, FINAL_BOTTLENECK_TOKENS):
            raise AssertionError(
                f"Final bottleneck shape mismatch: {tuple(final_channels_first.shape)}"
            )
        bottleneck = final_channels_first.transpose(1, 2)
        return bottleneck, {
            "audio_local": audio_local,
            "imu_local": imu_local,
            "audio": audio_features,
            "imu": imu_features,
            "multimodal": multimodal,
            "audio_to_imu": cross_attention["audio_to_imu"],
            "imu_to_audio": cross_attention["imu_to_audio"],
            "temporal_attention": temporal_attention,
            "cross_features": cross_features,
            "temporal_features": temporal_features,
            "recurrent_input": recurrent_input,
            "gru": sequence,
            "gru_hidden": hidden,
            "first_bottleneck": first.transpose(1, 2),
            "bottleneck": bottleneck,
        }

    def forward(
        self, audio: torch.Tensor, imu: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bottleneck, intermediates = self.encode(audio, imu)
        decoder_input = F.interpolate(
            bottleneck.transpose(1, 2),
            size=DECODER_INPUT_TOKENS,
            mode=self.decoder.upsample_mode,
        ).transpose(1, 2)
        if tuple(decoder_input.shape[1:]) != (DECODER_INPUT_TOKENS, 32):
            raise AssertionError(f"Decoder adapter shape mismatch: {tuple(decoder_input.shape)}")
        reconstruction = self.decoder(decoder_input)
        if return_attention:
            return reconstruction, {**intermediates, "decoder_input": decoder_input}
        return reconstruction


def temporal_mean_embedding(
    model: TrainingModel, audio: torch.Tensor, imu: torch.Tensor
) -> torch.Tensor:
    """Return the 32-D temporal mean of the 23 x 32 bottleneck."""

    bottleneck, _ = model.encode(audio, imu)
    if tuple(bottleneck.shape[1:]) != (FINAL_BOTTLENECK_TOKENS, EMBEDDING_DIMENSION):
        raise AssertionError(f"Bottleneck contract changed: {tuple(bottleneck.shape)}")
    embedding = torch.mean(bottleneck, dim=1)
    if tuple(embedding.shape[1:]) != (EMBEDDING_DIMENSION,):
        raise AssertionError(f"Embedding shape mismatch: {tuple(embedding.shape)}")
    return embedding
