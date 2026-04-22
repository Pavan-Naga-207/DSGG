import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import timm
from timm.data import resolve_data_config


@dataclass(frozen=True)
class DINOv2VisualConfig:
    model_name: str
    input_size: int
    train_lsj_min: float
    train_lsj_max: float
    patch_size: int
    detector_stride: int
    interpolation_name: str
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    pad_rgb: Tuple[float, float, float]
    dynamic_img_size: bool
    dynamic_img_pad: bool
    freeze_backbone: bool
    override_summary: str

    @property
    def interpolation(self):
        return cv2_interpolation_from_name(self.interpolation_name)

    @property
    def expected_base_hw(self):
        size = int((self.input_size + self.detector_stride - 1) / self.detector_stride)
        return size, size


def cv2_interpolation_from_name(name: str) -> int:
    key = str(name).strip().lower()
    if key in ('bicubic', 'cubic'):
        return cv2.INTER_CUBIC
    if key in ('bilinear', 'linear'):
        return cv2.INTER_LINEAR
    if key in ('nearest',):
        return cv2.INTER_NEAREST
    if key in ('lanczos', 'lanczos4'):
        return cv2.INTER_LANCZOS4
    if key in ('area',):
        return cv2.INTER_AREA
    return cv2.INTER_CUBIC


def _parse_rgb_triplet(raw_value: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if raw_value is None:
        return None
    values = [float(token.strip()) for token in str(raw_value).split(',') if token.strip()]
    if len(values) != 3:
        raise ValueError(
            'Expected an RGB triplet like "0.485,0.456,0.406", got {}'.format(raw_value)
        )
    return tuple(values)


@lru_cache(maxsize=8)
def _resolve_base_timm_cfg(model_name: str):
    backbone = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )
    try:
        data_cfg = resolve_data_config({}, model=backbone)
    except TypeError:
        data_cfg = resolve_data_config(backbone.pretrained_cfg)
    patch_size = backbone.patch_embed.patch_size
    patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else int(patch_size)
    return dict(data_cfg), patch_size


def resolve_dinov2_visual_config(
    model_name: str = 'vit_base_patch14_dinov2',
    input_size: int = 1024,
    train_lsj_min: float = 0.5,
    train_lsj_max: float = 2.0,
    freeze_backbone: bool = True,
    detector_stride: int = 16,
    interpolation_name: Optional[str] = None,
    pad_rgb: Optional[Tuple[float, float, float]] = None,
    dynamic_img_size: bool = True,
    dynamic_img_pad: bool = True,
) -> DINOv2VisualConfig:
    data_cfg, patch_size = _resolve_base_timm_cfg(model_name)
    mean = tuple(float(x) for x in data_cfg.get('mean', (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in data_cfg.get('std', (0.229, 0.224, 0.225)))
    interpolation_name = interpolation_name or str(data_cfg.get('interpolation', 'bicubic'))
    pad_rgb = pad_rgb or mean

    override_notes = []
    if input_size % patch_size != 0:
        override_notes.append(
            'input_size={} not divisible by patch_size={}; dynamic_img_pad={}'.format(
                input_size, patch_size, dynamic_img_pad
            )
        )
    if interpolation_name != str(data_cfg.get('interpolation', 'bicubic')):
        override_notes.append('interpolation_override={}'.format(interpolation_name))
    if pad_rgb != mean:
        override_notes.append('pad_rgb_override={}'.format(pad_rgb))

    return DINOv2VisualConfig(
        model_name=model_name,
        input_size=int(input_size),
        train_lsj_min=float(train_lsj_min),
        train_lsj_max=float(train_lsj_max),
        patch_size=int(patch_size),
        detector_stride=int(detector_stride),
        interpolation_name=str(interpolation_name),
        mean=mean,
        std=std,
        pad_rgb=tuple(float(x) for x in pad_rgb),
        dynamic_img_size=bool(dynamic_img_size),
        dynamic_img_pad=bool(dynamic_img_pad),
        freeze_backbone=bool(freeze_backbone),
        override_summary='; '.join(override_notes) if override_notes else 'none',
    )


@lru_cache(maxsize=1)
def get_phase3_dinov2_config() -> DINOv2VisualConfig:
    return resolve_dinov2_visual_config(
        model_name=os.environ.get('PHASE3_DINOV2_MODEL', 'vit_base_patch14_dinov2'),
        input_size=int(os.environ.get('PHASE3_INPUT_SIZE', '1024')),
        train_lsj_min=float(os.environ.get('PHASE3_LSJ_MIN', '0.5')),
        train_lsj_max=float(os.environ.get('PHASE3_LSJ_MAX', '2.0')),
        freeze_backbone=os.environ.get('PHASE3_FREEZE_BACKBONE', '1') == '1',
        detector_stride=int(os.environ.get('PHASE3_DETECTOR_STRIDE', '16')),
        interpolation_name=os.environ.get('PHASE3_INTERPOLATION', None),
        pad_rgb=_parse_rgb_triplet(os.environ.get('PHASE3_PAD_RGB', None)),
        dynamic_img_size=os.environ.get('PHASE3_DYNAMIC_IMG_SIZE', '1') == '1',
        dynamic_img_pad=os.environ.get('PHASE3_DYNAMIC_IMG_PAD', '1') == '1',
    )


def format_visual_config(cfg: DINOv2VisualConfig) -> str:
    return (
        'model={} target_size={} lsj=[{}, {}] interpolation={} mean={} std={} pad_rgb={} '
        'patch={} detector_stride={} dynamic_img_size={} dynamic_img_pad={} freeze_backbone={} overrides={}'
    ).format(
        cfg.model_name,
        cfg.input_size,
        cfg.train_lsj_min,
        cfg.train_lsj_max,
        cfg.interpolation_name,
        cfg.mean,
        cfg.std,
        cfg.pad_rgb,
        cfg.patch_size,
        cfg.detector_stride,
        cfg.dynamic_img_size,
        cfg.dynamic_img_pad,
        cfg.freeze_backbone,
        cfg.override_summary,
    )
