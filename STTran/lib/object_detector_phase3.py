import os
import sys
from typing import Optional

import torch

_special_topics_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _special_topics_root not in sys.path:
    sys.path.insert(0, _special_topics_root)

from lib.object_detector import detector as LegacyDetector
from phase3_dinov2.dinov2_bridge import DINOv2Backbone
from phase3_dinov2.preprocess import format_visual_config, get_phase3_dinov2_config


class detector(LegacyDetector):
    """
    Phase 3 detector wrapper: keeps the working Faster R-CNN/STTran shell, but
    swaps the visual backbone path to a frozen DINOv2 bridge.
    """

    def __init__(
        self,
        train,
        object_classes,
        use_SUPPLY,
        mode='predcls',
        backbone='dinov2',
        det_threshold=0.1,
    ):
        requested_backbone = str(backbone).lower()
        if requested_backbone != 'dinov2':
            raise ValueError('object_detector_phase3 expects backbone=dinov2, got {}'.format(backbone))
        self.phase3_requested_backbone = requested_backbone
        self.phase3_visual_cfg = get_phase3_dinov2_config()
        self.phase3_assert_path = os.environ.get('PHASE3_ASSERT_DINOV2_PATH', '1') == '1'
        self._phase3_entry_logged_modes = set()
        super().__init__(
            train=train,
            object_classes=object_classes,
            use_SUPPLY=use_SUPPLY,
            mode=mode,
            backbone='vitdet',
            det_threshold=det_threshold,
        )
        self.phase3_detector_shell = 'fasterrcnn'
        self.phase3_bridge_name = 'DINOv2Backbone'
        self.vitdet = DINOv2Backbone(
            model_name=self.phase3_visual_cfg.model_name,
            out_channels=int(os.environ.get('PHASE3_OUT_CHANNELS', '256')),
            align_channels=int(os.environ.get('PHASE3_ALIGN_CHANNELS', '1024')),
            freeze_backbone=self.phase3_visual_cfg.freeze_backbone,
        )
        self._log_init_summary()

    def _assert_phase3_path(self):
        if self.phase3_assert_path and not isinstance(self.vitdet, DINOv2Backbone):
            raise RuntimeError('Phase3 path assertion failed: expected DINOv2Backbone, got {}'.format(type(self.vitdet).__name__))

    def _trainable_param_count(self, module) -> int:
        return sum(param.numel() for param in module.parameters() if param.requires_grad)

    def _log_init_summary(self):
        self._assert_phase3_path()
        backbone_total = sum(param.numel() for param in self.vitdet.bridge.backbone.parameters())
        backbone_frozen = sum(
            param.numel() for param in self.vitdet.bridge.backbone.parameters() if not param.requires_grad
        )
        bridge_trainable = sum(
            param.numel()
            for name, param in self.vitdet.named_parameters()
            if param.requires_grad and not name.startswith('bridge.backbone.')
        )
        detector_trainable = self._trainable_param_count(self.fasterRCNN)
        shell_trainable = detector_trainable
        print('Phase3 detector init: backbone=dinov2 mode={} frozen_backbone={} detector_shell={}'.format(
            self.mode,
            self.phase3_visual_cfg.freeze_backbone,
            self.phase3_detector_shell,
        ))
        print(
            'Phase3 detector params: backbone frozen/total={} / {}  trainable_bridge={}  trainable_detector_shell={}'.format(
            backbone_frozen,
            backbone_total,
            bridge_trainable,
            shell_trainable,
        ))
        print('Phase3 detector visual cfg: {}'.format(format_visual_config(self.phase3_visual_cfg)))

    def _label_source_for_mode(self) -> str:
        if self.mode == 'predcls':
            return 'gt'
        if self.mode == 'sgcls':
            return os.environ.get('SGCLS_LABEL_SOURCE', 'detector')
        if self.mode == 'sgdet':
            return 'detector'
        return 'detector'

    def _set_phase3_runtime_context(self, mode_tag: str):
        self._assert_phase3_path()
        self.vitdet.set_runtime_context(mode_tag=mode_tag, label_source=self._label_source_for_mode())

    def _extract_base_features(self, images):
        self._assert_phase3_path()
        feat_dict = self.vitdet(images)
        if 'base' not in feat_dict:
            raise RuntimeError('Phase3 DINOv2 bridge did not return a base feature map')
        return feat_dict['base']

    def _log_entry_once(self, mode_tag: str, entry: Optional[dict]):
        if mode_tag in self._phase3_entry_logged_modes:
            return
        self._phase3_entry_logged_modes.add(mode_tag)
        if not isinstance(entry, dict):
            print('Phase3 entry [{}]: non-dict output {}'.format(mode_tag, type(entry).__name__))
            return
        boxes = entry.get('boxes')
        features = entry.get('features')
        fmaps = entry.get('fmaps')
        box_count = int(boxes.shape[0]) if torch.is_tensor(boxes) else -1
        feature_shape = tuple(features.shape) if torch.is_tensor(features) else None
        fmap_shape = tuple(fmaps.shape) if torch.is_tensor(fmaps) else None
        print(
            'Phase3 entry [{}]: backbone=dinov2 bridge={} label_source={} boxes_entering_sttran={} feature_shape={} fmap_shape={}'.format(
                mode_tag,
                self.phase3_bridge_name,
                self._label_source_for_mode(),
                box_count,
                feature_shape,
                fmap_shape,
            )
        )

    def forward_detector_pretrain(self, im_data, im_info, gt_boxes, num_boxes):
        self._set_phase3_runtime_context('detector_stage1')
        return super().forward_detector_pretrain(im_data, im_info, gt_boxes, num_boxes)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):
        self._set_phase3_runtime_context(self.mode)
        entry = super().forward(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all)
        self._log_entry_once(self.mode, entry)
        return entry
