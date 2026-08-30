"""Complete-graph, non-causal temporal attention."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalAttention(nn.Module):
    """Attend globally among all configured temporal tokens."""

    def __init__(
        self, token_count: int = 90, feature_dim: int = 48, negative_slope: float = 0.2
    ) -> None:
        super().__init__()
        self.timestamp_count = int(token_count)
        self.feature_dim = int(feature_dim)
        self.attention_vector = nn.Parameter(torch.empty(2 * self.feature_dim))
        self.negative_slope = float(negative_slope)
        nn.init.xavier_uniform_(self.attention_vector.view(1, -1))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.timestamp_count, self.feature_dim)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                f"TemporalAttention expected Bx{expected[0]}x{expected[1]}, "
                f"got {tuple(values.shape)}"
            )
        receiver_weights, source_weights = self.attention_vector.chunk(2)
        receiver_scores = torch.einsum("btd,d->bt", values, receiver_weights)
        source_scores = torch.einsum("btd,d->bt", values, source_weights)
        pairwise_scores = F.leaky_relu(
            receiver_scores.unsqueeze(-1) + source_scores.unsqueeze(-2),
            negative_slope=self.negative_slope,
        )
        attention = torch.softmax(pairwise_scores, dim=-1)
        output = torch.sigmoid(torch.bmm(attention, values))
        if output.shape != values.shape:
            raise AssertionError(f"Temporal attention shape mismatch: {tuple(output.shape)}")
        return output, attention
