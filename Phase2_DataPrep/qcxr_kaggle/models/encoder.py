"""
models/encoder.py
Frozen Swin Transformer visual encoder.
Works for both Swin-Tiny (local) and Swin-Base (Kaggle).
"""
import torch
import torch.nn as nn
from transformers import SwinModel, AutoFeatureExtractor


class FrozenSwinEncoder(nn.Module):
    """
    Wraps a HuggingFace SwinModel and keeps all parameters frozen.
    Returns last_hidden_state: [B, N_patches, hidden_dim]
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model = SwinModel.from_pretrained(model_name)
        # Freeze every parameter
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        # Expose hidden dim for downstream bottleneck
        self.hidden_dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, 224, 224]
        Returns:
            features: [B, N_patches, hidden_dim]
        """
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state  # [B, N, D]


def get_transforms(split: str):
    """Standard ImageNet normalisation transforms used by R2Gen."""
    from torchvision import transforms
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
