import math
from typing import Dict, Tuple

import timm
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn, Tensor
import torch.nn.functional as F


class SimpleViTDetFPN(nn.Module):
    """
    Minimal ViTDet-style bridge that turns flat ViT patch tokens into a
    synthetic feature pyramid compatible with downstream RoI heads.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        out_channels: int = 256,
        align_channels: int = 2048,
        freeze_backbone: bool = True,
        use_align: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.out_channels = out_channels
        self.align_channels = align_channels
        self.use_align = use_align

        # Backbone: MAE/ViT pretrain, classification head removed.
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        if not hasattr(self.backbone, "patch_embed"):
            raise RuntimeError(f"{model_name} missing patch_embed; expected a ViT-like model.")

        patch_size = self.backbone.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else int(patch_size)
        self.embed_dim = getattr(self.backbone, "embed_dim", None)
        if self.embed_dim is None:
            raise RuntimeError(f"{model_name} missing embed_dim; cannot build bridge.")
        self._base_grid_size = tuple(getattr(self.backbone.patch_embed, "grid_size", (14, 14)))
        self.register_buffer(
            "_base_pos_embed",
            self.backbone.pos_embed.detach().clone() if hasattr(self.backbone, "pos_embed") else torch.empty(0),
            persistent=False,
        )

        # Normalization to match ViT pretraining stats.
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1), persistent=False)

        # Simple 1x1 alignment to the expected detector head depth.
        if self.use_align:
            self.align_layers = nn.ModuleDict(
                {level: nn.Conv2d(out_channels, align_channels, kernel_size=1) for level in ("p2", "p3", "p4", "p5")}
            )
        else:
            self.align_layers = None

        # FPN heads: start from the 1/16 grid; upsample for p2, lateral for p3,
        # downsample for p4/p5. Keep channel count small to manage H100 memory.
        self.p2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, out_channels, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels, kernel_size=1),
            nn.GELU(),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _reshape_tokens(self, tokens: Tensor, height: int, width: int) -> Tensor:
        """
        tokens: [B, N, C] without CLS. Convert to [B, C, H/patch, W/patch].
        """
        b, n, c = tokens.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        if grid_h * grid_w != n:
            # Fall back to square inference from token count.
            side = int(math.sqrt(n))
            if side * side != n:
                raise RuntimeError(f"Token count {n} not square; cannot reshape to grid.")
            grid_h = grid_w = side
        return tokens.transpose(1, 2).reshape(b, c, grid_h, grid_w)

    def _resize_abs_pos_embed(
        self,
        source_pos_embed: Tensor,
        num_prefix_tokens: int,
        old_grid_size: Tuple[int, int],
        new_grid_size: Tuple[int, int],
    ) -> Tensor:
        if source_pos_embed.numel() == 0 or old_grid_size == new_grid_size:
            return source_pos_embed

        prefix = source_pos_embed[:, :num_prefix_tokens]
        grid = source_pos_embed[:, num_prefix_tokens:]
        grid = grid.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=new_grid_size, mode='bicubic', align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid_size[0] * new_grid_size[1], -1)
        return torch.cat([prefix, grid], dim=1)

    def forward(self, images: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Args:
            images: float tensor [B, 3, H, W] in range 0-1 or 0-255.
        Returns:
            aligned: dict of aligned feature maps (p2..p5) with channel=align_channels when use_align is True.
            raw: dict of raw pyramid maps (channel=out_channels).
        """
        if images.dtype != torch.float32:
            images = images.float()
        x = (images - self.pixel_mean) / self.pixel_std

        h, w = images.shape[2], images.shape[3]
        # Allow variable input sizes by updating patch_embed metadata.
        if hasattr(self.backbone, "patch_embed"):
            pe = self.backbone.patch_embed
            if hasattr(pe, "img_size"):
                pe.img_size = (h, w)
            if hasattr(pe, "grid_size"):
                pe.grid_size = (h // self.patch_size, w // self.patch_size)
            if hasattr(pe, "num_patches"):
                pe.num_patches = pe.grid_size[0] * pe.grid_size[1]

        # Resize positional embeddings if the token count changes.
        if hasattr(self.backbone, "pos_embed"):
            pos_embed = self.backbone.pos_embed
            num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
            num_patches = getattr(self.backbone.patch_embed, "num_patches", None)
            if num_patches is not None:
                target_tokens = num_prefix + num_patches
                if pos_embed.shape[1] != target_tokens:
                    source = self._base_pos_embed.to(
                        device=pos_embed.device,
                        dtype=pos_embed.dtype,
                    )
                    resized = self._resize_abs_pos_embed(
                        source_pos_embed=source,
                        num_prefix_tokens=num_prefix,
                        old_grid_size=self._base_grid_size,
                        new_grid_size=self.backbone.patch_embed.grid_size,
                    )
                    self.backbone.pos_embed = nn.Parameter(resized)

        features = self.backbone.forward_features(x)
        tokens = None
        if isinstance(features, torch.Tensor):
            tokens = features
        elif isinstance(features, dict):
            tokens = features.get("x") or features.get("last_hidden_state")
        elif isinstance(features, (tuple, list)):
            tokens = features[-1]
        if tokens is None or tokens.dim() != 3:
            raise RuntimeError(f"Unexpected feature output from {self.model_name}: {type(features)}")

        # Drop CLS token if present.
        # Drop prefix tokens (e.g., CLS). Timm ViTs usually expose num_prefix_tokens and
        # patch_embed.num_patches.
        num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        num_patches = getattr(self.backbone.patch_embed, "num_patches", None)
        expected = None
        if num_patches is not None:
            expected = num_prefix + num_patches
        if expected is not None and tokens.shape[1] == expected:
            tokens = tokens[:, num_prefix:, :]

        base = self._reshape_tokens(tokens, h, w)  # [B, C, H/16, W/16]

        p3 = self.p3(base)
        p2 = self.p2(base)
        p4 = self.p4(base)
        p5 = self.p5(p4)

        raw = {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
        if not self.use_align or self.align_layers is None:
            return raw, raw

        aligned = {level: self.align_layers[level](feat) for level, feat in raw.items()}
        return aligned, raw


def build_vitdet_bridge(
    model_name: str = "vit_base_patch16_224",
    out_channels: int = 256,
    align_channels: int = 2048,
    freeze_backbone: bool = True,
    use_align: bool = True,
) -> SimpleViTDetFPN:
    """
    Convenience factory to mirror the existing detector build pattern.
    """
    return SimpleViTDetFPN(
        model_name=model_name,
        out_channels=out_channels,
        align_channels=align_channels,
        freeze_backbone=freeze_backbone,
        use_align=use_align,
    )

