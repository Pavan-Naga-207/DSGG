import math
import os
from typing import Dict, Tuple

import timm
import torch
from timm.data import resolve_data_config
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

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        try:
            self.data_cfg = resolve_data_config({}, model=self.backbone)
        except TypeError:
            self.data_cfg = resolve_data_config(self.backbone.pretrained_cfg)
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
        self.default_input_size = int(os.environ.get("VIT_INPUT_SIZE", "0"))
        if self.default_input_size > 0:
            self.default_input_size = int(
                math.ceil(float(self.default_input_size) / float(self.patch_size)) * self.patch_size
            )

        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.data_cfg.get("mean", (0.5, 0.5, 0.5))).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(self.data_cfg.get("std", (0.5, 0.5, 0.5))).view(1, 3, 1, 1),
            persistent=False,
        )


        if self.use_align:
            self.align_layers = nn.ModuleDict(
                {level: nn.Conv2d(out_channels, align_channels, kernel_size=1) for level in ("p2", "p3", "p4", "p5")}
            )
        else:
            self.align_layers = None


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

        if self.default_input_size > 0:
            self._set_backbone_grid(self.default_input_size, self.default_input_size)
            self._maybe_resize_pos_embed()

    def _reshape_tokens(self, tokens: Tensor, height: int, width: int) -> Tensor:

        b, n, c = tokens.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        if grid_h * grid_w != n:

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

    def _set_backbone_grid(self, height: int, width: int) -> None:
        if not hasattr(self.backbone, "patch_embed"):
            return
        pe = self.backbone.patch_embed
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        if hasattr(pe, "img_size"):
            pe.img_size = (height, width)
        if hasattr(pe, "grid_size"):
            pe.grid_size = (grid_h, grid_w)
        if hasattr(pe, "num_patches"):
            pe.num_patches = grid_h * grid_w

    def _maybe_resize_pos_embed(self) -> None:
        if not hasattr(self.backbone, "pos_embed"):
            return
        pos_embed = self.backbone.pos_embed
        num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        num_patches = getattr(self.backbone.patch_embed, "num_patches", None)
        if num_patches is None:
            return
        target_tokens = num_prefix + num_patches
        if pos_embed.shape[1] == target_tokens:
            return
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

    def forward(self, images: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        if images.dtype != torch.float32:
            images = images.float()
       
        needs_rescale = (images.amax(dim=(1, 2, 3), keepdim=True) > 1.5).to(images.dtype)
        images = images / (1.0 + needs_rescale * 254.0)
        x = (images - self.pixel_mean) / self.pixel_std

        h, w = images.shape[2], images.shape[3]

        self._set_backbone_grid(h, w)

        self._maybe_resize_pos_embed()

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

    return SimpleViTDetFPN(
        model_name=model_name,
        out_channels=out_channels,
        align_channels=align_channels,
        freeze_backbone=freeze_backbone,
        use_align=use_align,
    )
