import math
import os
from typing import Dict, Optional

import timm
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from phase3_dinov2.preprocess import (
    format_visual_config,
    get_phase3_dinov2_config,
    resolve_dinov2_visual_config,
)


class DINOv2Bridge(nn.Module):
    """
    Frozen plain-ViT DINOv2 bridge that exposes a detector-friendly synthetic
    pyramid while keeping model-native photometric preprocessing.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        out_channels: Optional[int] = None,
        align_channels: Optional[int] = None,
        freeze_backbone: Optional[bool] = None,
        detector_stride: Optional[int] = None,
        use_align: bool = True,
    ):
        super().__init__()
        cfg = get_phase3_dinov2_config()
        if (
            model_name is not None
            or out_channels is not None
            or align_channels is not None
            or freeze_backbone is not None
            or detector_stride is not None
        ):
            cfg = resolve_dinov2_visual_config(
                model_name=model_name or cfg.model_name,
                input_size=cfg.input_size,
                train_lsj_min=cfg.train_lsj_min,
                train_lsj_max=cfg.train_lsj_max,
                freeze_backbone=cfg.freeze_backbone if freeze_backbone is None else bool(freeze_backbone),
                detector_stride=cfg.detector_stride if detector_stride is None else int(detector_stride),
                interpolation_name=cfg.interpolation_name,
                pad_rgb=cfg.pad_rgb,
                dynamic_img_size=cfg.dynamic_img_size,
                dynamic_img_pad=cfg.dynamic_img_pad,
            )
        self.visual_cfg = cfg
        self.model_name = cfg.model_name
        self.out_channels = int(
            out_channels if out_channels is not None else os.environ.get('PHASE3_OUT_CHANNELS', '256')
        )
        self.align_channels = int(
            align_channels if align_channels is not None else os.environ.get('PHASE3_ALIGN_CHANNELS', '1024')
        )
        self.use_align = bool(use_align)
        self.freeze_backbone = bool(cfg.freeze_backbone if freeze_backbone is None else freeze_backbone)
        self.detector_stride = int(cfg.detector_stride)

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=cfg.dynamic_img_size,
            dynamic_img_pad=cfg.dynamic_img_pad,
        )
        if not hasattr(self.backbone, 'patch_embed'):
            raise RuntimeError('{} is missing patch_embed; expected a ViT-like model'.format(self.model_name))
        patch_size = self.backbone.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else int(patch_size)
        self.embed_dim = int(getattr(self.backbone, 'embed_dim', 0))
        self.num_prefix_tokens = int(getattr(self.backbone, 'num_prefix_tokens', 1))
        if self.embed_dim <= 0:
            raise RuntimeError('{} missing embed_dim; cannot build DINOv2 bridge'.format(self.model_name))

        self.register_buffer(
            'pixel_mean',
            torch.tensor(cfg.mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'pixel_std',
            torch.tensor(cfg.std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.p2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.out_channels, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=1),
            nn.GELU(),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        if self.use_align:
            self.align_layers = nn.ModuleDict(
                {
                    level: nn.Conv2d(self.out_channels, self.align_channels, kernel_size=1)
                    for level in ('p2', 'p3', 'p4', 'p5')
                }
            )
        else:
            self.align_layers = None

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            self.backbone.eval()

        self._runtime_context = {'mode_tag': 'unset', 'label_source': 'unset'}
        self._runtime_logged_modes = set()
        self._startup_summary_printed = False
        self._print_startup_summary()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def set_runtime_context(self, mode_tag: str, label_source: str = 'unset'):
        self._runtime_context = {
            'mode_tag': str(mode_tag),
            'label_source': str(label_source),
        }

    def _print_startup_summary(self):
        if self._startup_summary_printed:
            return
        self._startup_summary_printed = True
        total_params = sum(param.numel() for param in self.backbone.parameters())
        frozen_params = sum(param.numel() for param in self.backbone.parameters() if not param.requires_grad)
        expected_h, expected_w = self.visual_cfg.expected_base_hw
        print('Phase3 DINOv2 bridge init:')
        print('  model={} embed_dim={} patch={} detector_stride={}'.format(
            self.model_name,
            self.embed_dim,
            self.patch_size,
            self.detector_stride,
        ))
        print('  expected base feature shape for {} input: (1, {}, {}, {})'.format(
            self.visual_cfg.input_size,
            self.align_channels if self.use_align else self.out_channels,
            expected_h,
            expected_w,
        ))
        print('  frozen backbone params: {} / {}'.format(frozen_params, total_params))
        print('  preprocessing: {}'.format(format_visual_config(self.visual_cfg)))

    def _normalize_images(self, images: Tensor) -> Tensor:
        if images.dtype != torch.float32:
            images = images.float()
        max_per_image = images.amax(dim=(1, 2, 3), keepdim=True)
        needs_rescale = (max_per_image > 1.5).to(images.dtype)
        images = images / (1.0 + needs_rescale * 254.0)
        return (images - self.pixel_mean) / self.pixel_std

    def _grid_size(self, height: int, width: int):
        if self.visual_cfg.dynamic_img_pad:
            return (
                int(math.ceil(float(height) / float(self.patch_size))),
                int(math.ceil(float(width) / float(self.patch_size))),
            )
        return int(height // self.patch_size), int(width // self.patch_size)

    def _extract_tokens(self, features: Tensor) -> Tensor:
        tokens = features
        if tokens.dim() != 3:
            raise RuntimeError('Unexpected DINOv2 feature output shape: {}'.format(tuple(tokens.shape)))
        grid_h, grid_w = self._grid_size(int(self._input_hw[0]), int(self._input_hw[1]))
        expected_tokens = grid_h * grid_w
        if tokens.shape[1] == expected_tokens + self.num_prefix_tokens:
            tokens = tokens[:, self.num_prefix_tokens:, :]
        elif tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                'DINOv2 token count mismatch for input {}x{}: got {}, expected {} (+{} prefix)'.format(
                    int(self._input_hw[0]),
                    int(self._input_hw[1]),
                    tokens.shape[1],
                    expected_tokens,
                    self.num_prefix_tokens,
                )
            )
        return tokens

    def _tokens_to_grid(self, tokens: Tensor, height: int, width: int) -> Tensor:
        grid_h, grid_w = self._grid_size(height, width)
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w)

    def _resample_to_detector_stride(self, token_grid: Tensor, height: int, width: int) -> Tensor:
        target_h = int(math.ceil(float(height) / float(self.detector_stride)))
        target_w = int(math.ceil(float(width) / float(self.detector_stride)))
        if token_grid.shape[-2:] == (target_h, target_w):
            return token_grid
        return F.interpolate(token_grid, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def _log_runtime_once(self, base: Tensor):
        mode_tag = self._runtime_context.get('mode_tag', 'unset')
        label_source = self._runtime_context.get('label_source', 'unset')
        if mode_tag in self._runtime_logged_modes:
            return
        self._runtime_logged_modes.add(mode_tag)
        print(
            'Phase3 backbone path [{}]: backbone=dinov2 bridge={} feature_shape={} preprocessing={} frozen={} label_source={}'.format(
                mode_tag,
                self.__class__.__name__,
                tuple(base.shape),
                format_visual_config(self.visual_cfg),
                self.freeze_backbone,
                label_source,
            )
        )

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        normalized = self._normalize_images(images)
        height, width = int(normalized.shape[2]), int(normalized.shape[3])
        self._input_hw = (height, width)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.forward_features(normalized)
        else:
            features = self.backbone.forward_features(normalized)
        tokens = self._extract_tokens(features)
        token_grid = self._tokens_to_grid(tokens, height=height, width=width)
        base_grid = self._resample_to_detector_stride(token_grid, height=height, width=width)

        p3 = self.p3(base_grid)
        p2 = self.p2(base_grid)
        p4 = self.p4(base_grid)
        p5 = self.p5(p4)
        raw = {'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5}
        if self.align_layers is None:
            base = raw['p3']
            aligned = raw
        else:
            aligned = {level: self.align_layers[level](feat) for level, feat in raw.items()}
            base = aligned['p3']
        self._log_runtime_once(base)
        return {'base': base, **aligned, **{f'raw_{level}': feat for level, feat in raw.items()}}


class DINOv2Backbone(nn.Module):
    """
    Thin adapter mirroring the ViTDetBackbone interface expected by the detector.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        out_channels: Optional[int] = None,
        align_channels: Optional[int] = None,
        freeze_backbone: Optional[bool] = None,
    ):
        super().__init__()
        self.bridge = DINOv2Bridge(
            model_name=model_name,
            out_channels=out_channels,
            align_channels=align_channels,
            freeze_backbone=freeze_backbone,
            use_align=True,
        )

    @property
    def visual_cfg(self):
        return self.bridge.visual_cfg

    def set_runtime_context(self, mode_tag: str, label_source: str = 'unset'):
        self.bridge.set_runtime_context(mode_tag=mode_tag, label_source=label_source)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.bridge(images)
