"""Prediction models on top of SBERT sentence embeddings.

Default (PS-6): a three-layer 1D-CNN regressor (guidelines §3.1).
Optional: a 2-layer MLP sanity baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNRegressor(nn.Module):
    """3-layer Conv1D regressor over the SBERT sentence vector.

    The sentence embedding (e.g. 384-dim) is treated as a length-D single-channel
    sequence and passed through three Conv1D blocks (64 -> 128 -> 256 channels,
    kernel 3, BatchNorm + ReLU + Dropout), then global avg pool + FC head.
    """

    def __init__(self, input_dim: int, filters: list[int] = (64, 128, 256),
                 kernel_size: int = 3, dropout: float = 0.3, fc_dim: int = 64):
        super().__init__()
        blocks = []
        in_ch = 1
        for f in filters:
            blocks.append(nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=f,
                          kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(f),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ))
            in_ch = f
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_ch, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> (B, 1, D)
        x = x.unsqueeze(1)
        h = self.conv(x)              # (B, C, D)
        h = self.pool(h).squeeze(-1)  # (B, C)
        return self.head(h).squeeze(-1)  # (B,)


class MLPRegressor(nn.Module):
    """Sanity baseline: 2-layer MLP regressor (not the default for PS-6)."""

    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(cfg: dict, input_dim: int) -> nn.Module:
    arch = str(cfg.get("model", {}).get("arch", "cnn")).lower()
    if arch == "cnn":
        return CNNRegressor(
            input_dim=input_dim,
            filters=list(cfg["model"]["cnn_filters"]),
            kernel_size=int(cfg["model"]["kernel_size"]),
            dropout=float(cfg["model"]["dropout"]),
            fc_dim=int(cfg["model"]["fc_dim"]),
        )
    if arch == "mlp":
        return MLPRegressor(input_dim=input_dim,
                            dropout=float(cfg["model"]["dropout"]))
    raise ValueError(f"Unknown model.arch: {arch}")
