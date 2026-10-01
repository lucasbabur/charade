"""Networks. Both take (categorical ids [n, F], dense [n, D]) and return logits [n]."""

from collections.abc import Sequence

import torch
from torch import nn


class FieldEmbedding(nn.Module):
    """One table for all fields, addressed with per-field offsets (fast and ONNX-friendly)."""

    def __init__(self, cardinalities: list[int], dim: int) -> None:
        super().__init__()
        self.offsets: torch.Tensor
        self.register_buffer("offsets", torch.tensor([0, *cardinalities[:-1]]).cumsum(0), persistent=True)
        self.table = nn.Embedding(sum(cardinalities), dim)
        nn.init.normal_(self.table.weight, std=0.01)

    def forward(self, categorical: torch.Tensor) -> torch.Tensor:
        """[n, F] ids -> [n, F, dim]."""
        return self.table(categorical + self.offsets)


class Logistic(nn.Module):
    """Logistic regression on one-hot categoricals plus linear dense terms (the baseline)."""

    def __init__(self, cardinalities: list[int], n_dense: int) -> None:
        super().__init__()
        self.weights = FieldEmbedding(cardinalities, 1)
        nn.init.zeros_(self.weights.table.weight)
        self.dense = nn.Linear(n_dense, 1) if n_dense else None
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        """Sum of per-value weights, dense terms and bias."""
        logit = self.weights(categorical).sum(dim=(1, 2)) + self.bias
        if self.dense is not None:
            logit = logit + self.dense(dense).squeeze(-1)
        return logit


class CrossLayer(nn.Module):
    """DCN-v2 low-rank cross: x_{l+1} = x0 * (U V^T x_l + b) + x_l."""

    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.v = nn.Linear(width, rank, bias=False)
        self.u = nn.Linear(rank, width)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One explicit feature-crossing step."""
        return x0 * self.u(self.v(x)) + x


class DCNv2(nn.Module):
    """Deep & Cross Network v2 (Wang et al. 2021), stacked: embeddings -> cross layers -> MLP -> logit."""

    def __init__(
        self,
        cardinalities: list[int],
        n_dense: int,
        embedding_dim: int,
        cross_layers: int,
        cross_rank: int,
        hidden: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = FieldEmbedding(cardinalities, embedding_dim)
        width = len(cardinalities) * embedding_dim + n_dense
        self.cross = nn.ModuleList(CrossLayer(width, cross_rank) for _ in range(cross_layers))
        layers: list[nn.Module] = []
        previous = width
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)]
            previous = size
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(previous, 1)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        """Logit per row."""
        x0 = torch.cat([self.embedding(categorical).flatten(1), dense], dim=1)
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return self.head(self.deep(x)).squeeze(-1)


class Ensemble(nn.Module):
    """Mean logit of several seeds; exported as one ONNX graph."""

    def __init__(self, members: Sequence[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        """Average of member logits."""
        return torch.stack([m(categorical, dense) for m in self.members]).mean(0)
