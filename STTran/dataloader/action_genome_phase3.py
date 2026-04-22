import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader import action_genome as legacy_action_genome

_special_topics_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _special_topics_root not in sys.path:
    sys.path.insert(0, _special_topics_root)

from phase3_dinov2.preprocess import format_visual_config, get_phase3_dinov2_config

_round_up_to_multiple = legacy_action_genome._round_up_to_multiple
_resize_to_fit_square = legacy_action_genome._resize_to_fit_square
_pad_bottom_right = legacy_action_genome._pad_bottom_right
_build_detection_targets = legacy_action_genome._build_detection_targets
cuda_collate_fn = legacy_action_genome.cuda_collate_fn
LegacyAG = legacy_action_genome.AG


class AG(Dataset):
    """
    Phase 3 Action Genome dataset that keeps detector geometry but uses a
    DINOv2-native photometric config everywhere.
    """

    def __init__(
        self,
        mode,
        datasize,
        data_path=None,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
        backbone='dinov2',
        return_detection_targets=False,
        min_frames_per_video=3,
    ):
        super().__init__()
        self.mode = mode
        self.backbone = str(backbone).lower()
        if self.backbone != 'dinov2':
            raise ValueError('Phase3 AG expects backbone=dinov2, got {}'.format(backbone))
        self.return_detection_targets = bool(return_detection_targets)
        self.min_frames_per_video = int(min_frames_per_video)
        self.visual_cfg = get_phase3_dinov2_config()

        self._legacy = LegacyAG(
            mode=mode,
            datasize=datasize,
            data_path=data_path,
            filter_nonperson_box_frame=filter_nonperson_box_frame,
            filter_small_box=filter_small_box,
            backbone='resnet101',
            return_detection_targets=False,
            min_frames_per_video=min_frames_per_video,
        )

        self.frames_path = self._legacy.frames_path
        self.object_classes = self._legacy.object_classes
        self.relationship_classes = self._legacy.relationship_classes
        self.attention_relationships = self._legacy.attention_relationships
        self.spatial_relationships = self._legacy.spatial_relationships
        self.contacting_relationships = self._legacy.contacting_relationships
        self.video_list = self._legacy.video_list
        self.video_size = self._legacy.video_size
        self.gt_annotations = self._legacy.gt_annotations
        self.non_gt_human_nums = self._legacy.non_gt_human_nums
        self.non_heatmap_nums = self._legacy.non_heatmap_nums
        self.non_person_video = self._legacy.non_person_video
        self.short_video = self._legacy.short_video
        self.valid_nums = self._legacy.valid_nums

        self.dinov2_model_name = self.visual_cfg.model_name
        self.dinov2_input_size = int(self.visual_cfg.input_size)
        self.dinov2_patch_size = int(self.visual_cfg.patch_size)
        self.dinov2_interpolation_name = self.visual_cfg.interpolation_name
        self.dinov2_interpolation = self.visual_cfg.interpolation
        self.dinov2_mean = tuple(float(x) for x in self.visual_cfg.mean)
        self.dinov2_std = tuple(float(x) for x in self.visual_cfg.std)
        self.dinov2_pad_rgb = np.asarray(self.visual_cfg.pad_rgb, dtype=np.float32)

        # Compatibility attrs for detector-stage helpers that expect ViT-style names.
        self.vit_model_name = self.dinov2_model_name
        self.vit_input_size = self.dinov2_input_size
        self.vit_patch_size = self.dinov2_patch_size
        self.vit_interpolation_name = self.dinov2_interpolation_name
        self.vit_interpolation = self.dinov2_interpolation
        self.vit_pad_rgb = self.dinov2_pad_rgb
        self.vit_data_cfg = {
            'mean': self.dinov2_mean,
            'std': self.dinov2_std,
            'interpolation': self.dinov2_interpolation_name,
            'input_size': (3, self.dinov2_input_size, self.dinov2_input_size),
        }

        print('Phase3 DINOv2 input pipeline: RGB float[0,1], fit-longest-side={}, square-pad={}, patch={}'.format(
            self.dinov2_input_size,
            self.dinov2_input_size,
            self.dinov2_patch_size,
        ))
        print('Phase3 DINOv2 visual cfg: {}'.format(format_visual_config(self.visual_cfg)))

    def __len__(self):
        return len(self.video_list)

    def __getattr__(self, name):
        if name == '_legacy':
            raise AttributeError(name)
        return getattr(self._legacy, name)

    def __getitem__(self, index):
        frame_names = self.video_list[index]
        processed_ims = []
        im_infos = []

        for name in frame_names:
            frame_path = os.path.join(self.frames_path, name)
            image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError('Failed to read frame: {}'.format(frame_path))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            resized_image, resize_scale = _resize_to_fit_square(
                image,
                self.dinov2_input_size,
                self.dinov2_interpolation,
            )
            resized_h, resized_w = resized_image.shape[:2]
            padded_image = _pad_bottom_right(
                resized_image,
                self.dinov2_input_size,
                self.dinov2_input_size,
                self.dinov2_pad_rgb,
            )
            processed_ims.append(np.ascontiguousarray(padded_image))
            im_infos.append([resized_h, resized_w, resize_scale])

        blob = np.stack(processed_ims, axis=0)
        im_info = torch.from_numpy(np.asarray(im_infos, dtype=np.float32))
        img_tensor = torch.from_numpy(np.ascontiguousarray(blob)).permute(0, 3, 1, 2)

        if self.return_detection_targets:
            gt_boxes, num_boxes = _build_detection_targets(self.gt_annotations[index], im_infos)
        else:
            gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5], dtype=torch.float32)
            num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index
