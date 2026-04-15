"""
models/bottleneck.py
Three classical visual bottleneck modules (Stage A baselines).
All map: [B, N_patches, visual_dim] → [B, 1, llm_dim]
  ─ Linear Bottleneck
  ─ MLP Bottleneck
  ─ Transformer Bottleneck
"""
import torch
import torch.nn as nn


class LinearBottleneck(nn.Module):
    """
    Simplest possible bottleneck: mean-pool patches → single linear projection.
    Trainable params: visual_dim × llm_dim
    """
    def __init__(self, visual_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Linear(visual_dim, llm_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, N, D]
        pooled = features.mean(dim=1)        # [B, D]
        out    = self.proj(pooled)            # [B, llm_dim]
        return out.unsqueeze(1)              # [B, 1, llm_dim]


class MLPBottleneck(nn.Module):
    """
    Two-layer MLP with GELU activation.
    Trainable params ≈ 2× LinearBottleneck
    """
    def __init__(self, visual_dim: int, llm_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=1)        # [B, D]
        out    = self.mlp(pooled)            # [B, llm_dim]
        return out.unsqueeze(1)              # [B, 1, llm_dim]


class TransformerBottleneck(nn.Module):
    """
    Lightweight Transformer encoder over patch tokens, then mean-pool + project.
    Adds cross-patch attention before pooling.
    """
    def __init__(self, visual_dim: int, llm_dim: int,
                 nhead: int = 8, num_layers: int = 2):
        super().__init__()
        # Swin-Tiny dim=768 is divisible by 8; Swin-Base dim=1024 is too.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=visual_dim, nhead=nhead,
            dim_feedforward=visual_dim * 2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.proj = nn.Linear(visual_dim, llm_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, N, D]
        attended = self.transformer(features)   # [B, N, D]
        pooled   = attended.mean(dim=1)         # [B, D]
        out      = self.proj(pooled)            # [B, llm_dim]
        return out.unsqueeze(1)                # [B, 1, llm_dim]


def get_bottleneck(name: str, visual_dim: int, llm_dim: int,
                   nhead: int = 8, trans_layers: int = 2) -> nn.Module:
    """Factory function."""
    name = name.lower()
    if name == "linear":
        return LinearBottleneck(visual_dim, llm_dim)
    elif name == "mlp":
        return MLPBottleneck(visual_dim, llm_dim)
    elif name == "transformer":
        return TransformerBottleneck(visual_dim, llm_dim, nhead, trans_layers)
    else:
        raise ValueError(f"Unknown bottleneck: {name}. Choose linear/mlp/transformer")
