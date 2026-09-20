"""ArcFace and supervised contrastive objectives."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ArcMarginProduct(nn.Module):
    """Normalized classifier with an additive angular margin."""

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.35) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(embeddings), F.normalize(self.weight))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = self.cosine_logits(embeddings).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None or self.weight.shape[0] == 1:
            return cosine * self.scale
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(dtype=cosine.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """SupCon objective for a flattened batch containing both views of each source."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features must be [N,D] and labels must be [N]")
    if features.shape[0] < 2:
        return features.new_zeros(())
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
    logits_mask = ~identity
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives_per_anchor = positive_mask.sum(dim=1)
    valid = positives_per_anchor > 0
    if not valid.any():
        return features.new_zeros(())
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positives_per_anchor.clamp_min(1)
    return -mean_log_prob[valid].mean()
