from typing import Dict

import torch
from torch import nn, Tensor

from phase2_vitdet.simple_vitdet_fpn import build_vitdet_bridge


class ViTDetBackbone(nn.Module):
    """
    Thin adapter that exposes a C4-like feature map at stride 1/16.
    Returns a dict so future multi-scale heads can consume p2–p5.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        out_channels: int = 256,
        align_channels: int = 1024,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.bridge = build_vitdet_bridge(
            model_name=model_name,
            out_channels=out_channels,
            align_channels=align_channels,
            freeze_backbone=freeze_backbone,
            use_align=True,
        )

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        aligned, raw = self.bridge(images)
        base = aligned["p3"]  # stride 1/16, aligns with ResNet C4
        return {"base": base, **aligned, **{f"raw_{k}": v for k, v in raw.items()}}

