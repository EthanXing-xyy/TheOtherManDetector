"""Decoder-free 32-D embedding models for native and PSoC 6 execution.

Author: Ethan_Xing
License: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .components import AudioEncoder, IMUEncoder
from .cross_modal_attention import CrossModalAttention
from .temporal_attention import TemporalAttention
from .training_model import (
    EMBEDDING_DIMENSION,
    FINAL_BOTTLENECK_TOKENS,
    FIRST_BOTTLENECK_TOKENS,
    GLOBAL_TOKEN_COUNT,
    PHYSICAL_FRAME_COUNT,
    POOL_KERNEL,
    POOL_STRIDE,
)


GRU_CHUNK_SIZE = 10
GRU_CHUNK_COUNT = GLOBAL_TOKEN_COUNT // GRU_CHUNK_SIZE
RUNTIME_MAX_CONCAT_INPUTS = 10

if GLOBAL_TOKEN_COUNT % GRU_CHUNK_SIZE:
    raise AssertionError("The recurrent sequence must divide into fixed-size chunks")
if GRU_CHUNK_COUNT > RUNTIME_MAX_CONCAT_INPUTS:
    raise AssertionError("The deployment concat exceeds the runtime input-count limit")


class TanhGELU(nn.Module):
    """Converter-compatible tanh approximation of GELU."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Use a tanh-based GELU approximation to keep the graph compatible
        # with the target converter/runtime while preserving GELU-like behavior.
        cubic = value * value * value
        return 0.5 * value * (
            1.0 + torch.tanh(0.7978845608028654 * (value + 0.044715 * cubic))
        )


class EmbeddingModel(nn.Module):
    """Encoder-side network with exact GELU and a native GRU call."""

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
        self.activation: nn.Module = nn.GELU()

    def _downsample(self, features: torch.Tensor, channels: int) -> torch.Tensor:
        if tuple(features.shape[1:]) != (PHYSICAL_FRAME_COUNT, channels):
            raise ValueError(f"Unexpected local feature shape: {tuple(features.shape)}")
        pooled = self.temporal_downsample(features.transpose(1, 2)).transpose(1, 2)
        if tuple(pooled.shape[1:]) != (GLOBAL_TOKEN_COUNT, channels):
            raise AssertionError(f"Unexpected pooled feature shape: {tuple(pooled.shape)}")
        return pooled

    def build_recurrent_input(self, audio: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        audio_features = self._downsample(self.audio_branch(audio), 32)
        imu_features = self._downsample(self.imu_branch(imu), 16)
        multimodal = torch.cat((audio_features, imu_features), dim=-1)
        cross_features, _ = self.feature_gat(multimodal)
        temporal_features, _ = self.time_gat(multimodal)
        recurrent_input = torch.cat(
            (multimodal, cross_features, temporal_features), dim=-1
        )
        if tuple(recurrent_input.shape[1:]) != (GLOBAL_TOKEN_COUNT, 144):
            raise AssertionError(f"GRU input shape mismatch: {tuple(recurrent_input.shape)}")
        return recurrent_input

    def finish_embedding(self, sequence: torch.Tensor) -> torch.Tensor:
        if tuple(sequence.shape[1:]) != (GLOBAL_TOKEN_COUNT, 64):
            raise AssertionError(f"GRU output shape mismatch: {tuple(sequence.shape)}")
        first = self.activation(self.bottleneck1(sequence.transpose(1, 2)))
        if tuple(first.shape[1:]) != (48, FIRST_BOTTLENECK_TOKENS):
            raise AssertionError(f"First bottleneck shape mismatch: {tuple(first.shape)}")
        final = self.activation(self.bottleneck2(first)).transpose(1, 2)
        if tuple(final.shape[1:]) != (FINAL_BOTTLENECK_TOKENS, EMBEDDING_DIMENSION):
            raise AssertionError(f"Final bottleneck shape mismatch: {tuple(final.shape)}")
        embedding = torch.mean(final, dim=1)
        if tuple(embedding.shape[1:]) != (EMBEDDING_DIMENSION,):
            raise AssertionError(f"Embedding shape mismatch: {tuple(embedding.shape)}")
        return embedding

    def forward(self, audio: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        recurrent_input = self.build_recurrent_input(audio, imu)
        sequence, _ = self.gru(recurrent_input)
        return self.finish_embedding(sequence)


class ConverterCompatibleEmbeddingModel(EmbeddingModel):
    """PSoC 6 converter graph with primitive GELU and chunked recurrence."""

    sequence_length = GLOBAL_TOKEN_COUNT
    chunk_size = GRU_CHUNK_SIZE
    chunk_count = GRU_CHUNK_COUNT
    maximum_explicit_concat_inputs = GRU_CHUNK_COUNT

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.audio_branch.activation = TanhGELU()
        self.imu_branch.activation = TanhGELU()
        self.activation = TanhGELU()

    def _chunked_gru(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (GLOBAL_TOKEN_COUNT, 144):
            raise ValueError(
                f"Chunked GRU expected Bx{GLOBAL_TOKEN_COUNT}x144, got {tuple(inputs.shape)}"
            )
        # Process the recurrent sequence in fixed-size chunks to keep converted
        # CONCAT nodes within the runtime-supported input limit.
        chunks: list[torch.Tensor] = []
        hidden: torch.Tensor | None = None
        for start in range(0, GLOBAL_TOKEN_COUNT, GRU_CHUNK_SIZE):
            chunk = inputs[:, start : start + GRU_CHUNK_SIZE, :]
            if hidden is None:
                output, hidden = self.gru(chunk)
            else:
                output, hidden = self.gru(chunk, hidden)
            chunks.append(output)
        if len(chunks) != GRU_CHUNK_COUNT:
            raise AssertionError(f"GRU chunk count changed: {len(chunks)}")
        sequence = torch.cat(chunks, dim=1)
        if tuple(sequence.shape[1:]) != (GLOBAL_TOKEN_COUNT, 64):
            raise AssertionError(f"Chunk assembly shape mismatch: {tuple(sequence.shape)}")
        if hidden is None or tuple(hidden.shape) != (1, inputs.shape[0], 64):
            raise AssertionError("Final GRU hidden-state shape mismatch")
        return sequence, hidden

    def forward(self, audio: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        recurrent_input = self.build_recurrent_input(audio, imu)
        sequence, _ = self._chunked_gru(recurrent_input)
        return self.finish_embedding(sequence)


def load_encoder_state(
    model: EmbeddingModel,
    training_state: Mapping[str, torch.Tensor],
) -> EmbeddingModel:
    """Load only encoder-side weights and verify that every tensor is exact."""

    expected = set(model.state_dict())
    decoder_keys = {name for name in training_state if name.startswith("decoder.")}
    selected = {name: value for name, value in training_state.items() if name in expected}
    missing = sorted(expected.difference(selected))
    unexpected = sorted(set(training_state).difference(expected).difference(decoder_keys))
    if missing or unexpected or not decoder_keys:
        raise KeyError(
            f"Deployment state mismatch: missing={missing}, unexpected={unexpected}, "
            f"decoder_parameter_count={len(decoder_keys)}"
        )
    model.load_state_dict(selected, strict=True)
    model.eval()
    for name, value in model.state_dict().items():
        if not torch.equal(value, training_state[name]):
            raise AssertionError(f"Learned tensor changed during load: {name}")
    return model


def load_checkpoint_encoder(
    checkpoint_path: str | Path,
    *,
    converter_compatible: bool = True,
) -> EmbeddingModel:
    """Construct a decoder-free encoder from the reconstruction checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain its model configuration")
    model: EmbeddingModel
    if converter_compatible:
        model = ConverterCompatibleEmbeddingModel(config)
    else:
        model = EmbeddingModel(config)
    return load_encoder_state(model, checkpoint["model_state_dict"])
