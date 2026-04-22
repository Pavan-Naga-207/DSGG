import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader.action_genome_phase3 import AG
from dataloader.action_genome import (
    _pad_bottom_right,
    _resize_to_fit_square,
    _round_up_to_multiple,
)


class AGDetectorStage1(Dataset):
    """
    Phase 3 frame-level Action Genome detector dataset using DINOv2-native
    photometric config plus detector-style LSJ geometry.
    """

    def __init__(
        self,
        mode,
        datasize,
        data_path=None,
        backbone='dinov2',
        filter_small_box=True,
    ):
        super().__init__()
        self.mode = mode
        self.backbone = str(backbone).lower()
        if self.backbone != 'dinov2':
            raise ValueError('AGDetectorStage1 phase3 expects backbone=dinov2, got {}'.format(backbone))

        self.sequence_dataset = AG(
            mode=mode,
            datasize=datasize,
            data_path=data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=filter_small_box,
            backbone=backbone,
            return_detection_targets=False,
            min_frames_per_video=1,
        )
        self.visual_cfg = self.sequence_dataset.visual_cfg
        self.frames_path = self.sequence_dataset.frames_path
        self.object_classes = self.sequence_dataset.object_classes
        self.relationship_classes = self.sequence_dataset.relationship_classes
        self.attention_relationships = self.sequence_dataset.attention_relationships
        self.spatial_relationships = self.sequence_dataset.spatial_relationships
        self.contacting_relationships = self.sequence_dataset.contacting_relationships

        self.target_size = int(self.visual_cfg.input_size)
        self.patch_size = int(self.visual_cfg.patch_size)
        self.pad_rgb = np.asarray(self.visual_cfg.pad_rgb, dtype=np.float32)
        self.interpolation = self.visual_cfg.interpolation
        self.max_crop_retries = max(1, int(os.environ.get('PHASE3_LSJ_MAX_RETRIES', '10')))
        if mode == 'train':
            self.scale_min = float(os.environ.get('PHASE3_LSJ_MIN', str(self.visual_cfg.train_lsj_min)))
            self.scale_max = float(os.environ.get('PHASE3_LSJ_MAX', str(self.visual_cfg.train_lsj_max)))
        else:
            self.scale_min = 1.0
            self.scale_max = 1.0

        self.samples = []
        self.sample_class_ids = []
        self.class_box_hist = np.zeros((len(self.object_classes),), dtype=np.int64)
        self.class_frame_hist = np.zeros((len(self.object_classes),), dtype=np.int64)
        for frame_names, gt_video in zip(self.sequence_dataset.video_list, self.sequence_dataset.gt_annotations):
            for frame_name, frame_annotation in zip(frame_names, gt_video):
                boxes = self._annotation_to_boxes(frame_annotation)
                if boxes.shape[0] > 0:
                    class_ids = boxes[:, 4].astype(np.int64)
                    class_ids = class_ids[(class_ids > 0) & (class_ids < len(self.object_classes))]
                    unique_class_ids = tuple(sorted(set(int(cls) for cls in class_ids.tolist())))
                    if class_ids.size > 0:
                        self.class_box_hist[: len(self.object_classes)] += np.bincount(
                            class_ids,
                            minlength=len(self.object_classes),
                        )[: len(self.object_classes)]
                        self.class_frame_hist[list(unique_class_ids)] += 1
                else:
                    unique_class_ids = tuple()
                self.samples.append((frame_name, frame_annotation))
                self.sample_class_ids.append(unique_class_ids)

        print(
            'Phase3 AGDetectorStage1: {} frames, target_size={}, scale_jitter=[{}, {}]'.format(
                len(self.samples),
                self.target_size,
                self.scale_min,
                self.scale_max,
            )
        )

    def __len__(self):
        return len(self.samples)

    def build_inverse_sqrt_class_weights(self, background_weight=1.0):
        counts = self.class_box_hist.astype(np.float64)
        weights = np.ones((len(self.object_classes),), dtype=np.float32)
        valid_fg = counts[1:] > 0
        if np.any(valid_fg):
            fg_weights = np.ones_like(counts[1:], dtype=np.float64)
            fg_weights[valid_fg] = 1.0 / np.sqrt(counts[1:][valid_fg])
            fg_mean = float(fg_weights[valid_fg].mean())
            if fg_mean > 0.0:
                fg_weights = fg_weights / fg_mean
            weights[1:] = fg_weights.astype(np.float32)
        weights[0] = float(background_weight)
        return torch.as_tensor(weights, dtype=torch.float32)

    def build_repeat_factor_weights(self, threshold=0.01, max_repeat=4.0):
        frame_freq = self.class_frame_hist.astype(np.float64) / float(max(1, len(self.samples)))
        class_repeat = np.ones((len(self.object_classes),), dtype=np.float64)
        valid_fg = frame_freq[1:] > 0.0
        if np.any(valid_fg):
            fg_repeat = class_repeat[1:]
            repeat_values = np.sqrt(float(threshold) / np.maximum(frame_freq[1:][valid_fg], np.finfo(np.float64).eps))
            fg_repeat[valid_fg] = np.clip(repeat_values, 1.0, float(max_repeat))
            class_repeat[1:] = fg_repeat

        sample_weights = np.ones((len(self.samples),), dtype=np.float64)
        for sample_idx, class_ids in enumerate(self.sample_class_ids):
            if class_ids:
                sample_weights[sample_idx] = max(class_repeat[int(cls)] for cls in class_ids)
        sample_mean = float(sample_weights.mean())
        if sample_mean > 0.0:
            sample_weights = sample_weights / sample_mean

        return (
            torch.as_tensor(sample_weights, dtype=torch.double),
            torch.as_tensor(class_repeat, dtype=torch.float32),
        )

    def _annotation_to_boxes(self, frame_annotation):
        boxes = []
        if len(frame_annotation) > 0 and 'person_bbox' in frame_annotation[0]:
            person_boxes = np.asarray(frame_annotation[0]['person_bbox'], dtype=np.float32)
            if person_boxes.ndim == 1:
                person_boxes = person_boxes.reshape(1, -1)
            for bbox in person_boxes:
                if bbox.shape[0] >= 4:
                    boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], 1.0])

        for rel in frame_annotation[1:]:
            bbox = np.asarray(rel['bbox'], dtype=np.float32).reshape(-1)
            if bbox.shape[0] < 4:
                continue
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], float(rel['class'])])

        if not boxes:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
        valid = (
            np.isfinite(boxes).all(axis=1)
            & ((boxes[:, 2] - boxes[:, 0]) >= 1.0)
            & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
        )
        return boxes[valid]

    def _transform_with_lsj(self, image, boxes):
        jitter_scale = 1.0
        if self.mode == 'train' and self.scale_max > self.scale_min:
            jitter_scale = random.uniform(self.scale_min, self.scale_max)

        jitter_target = max(
            self.patch_size,
            _round_up_to_multiple(int(round(self.target_size * jitter_scale)), self.patch_size),
        )
        resized_image, resize_scale = _resize_to_fit_square(image, jitter_target, self.interpolation)
        transformed_boxes = boxes.copy()
        if transformed_boxes.shape[0] > 0:
            transformed_boxes[:, :4] *= resize_scale

        resized_h, resized_w = resized_image.shape[:2]
        crop_h = min(self.target_size, resized_h)
        crop_w = min(self.target_size, resized_w)
        if self.mode == 'train':
            offset_y = random.randint(0, max(0, resized_h - crop_h))
            offset_x = random.randint(0, max(0, resized_w - crop_w))
        else:
            offset_y = max(0, (resized_h - crop_h) // 2)
            offset_x = max(0, (resized_w - crop_w) // 2)

        cropped = resized_image[offset_y: offset_y + crop_h, offset_x: offset_x + crop_w]
        if transformed_boxes.shape[0] > 0:
            transformed_boxes[:, [0, 2]] -= float(offset_x)
            transformed_boxes[:, [1, 3]] -= float(offset_y)
            transformed_boxes[:, 0] = np.clip(transformed_boxes[:, 0], 0.0, float(crop_w))
            transformed_boxes[:, 2] = np.clip(transformed_boxes[:, 2], 0.0, float(crop_w))
            transformed_boxes[:, 1] = np.clip(transformed_boxes[:, 1], 0.0, float(crop_h))
            transformed_boxes[:, 3] = np.clip(transformed_boxes[:, 3], 0.0, float(crop_h))
            keep = (
                (transformed_boxes[:, 2] - transformed_boxes[:, 0]) >= 1.0
            ) & (
                (transformed_boxes[:, 3] - transformed_boxes[:, 1]) >= 1.0
            )
            transformed_boxes = transformed_boxes[keep]

        if crop_h < self.target_size or crop_w < self.target_size:
            cropped = _pad_bottom_right(cropped, self.target_size, self.target_size, self.pad_rgb)

        im_info = torch.tensor([float(crop_h), float(crop_w), float(resize_scale)], dtype=torch.float32)
        return cropped, transformed_boxes, im_info

    def _transform_keep_all(self, image, boxes):
        resized_image, resize_scale = _resize_to_fit_square(image, self.target_size, self.interpolation)
        transformed_boxes = boxes.copy()
        if transformed_boxes.shape[0] > 0:
            transformed_boxes[:, :4] *= resize_scale

        resized_h, resized_w = resized_image.shape[:2]
        if resized_h < self.target_size or resized_w < self.target_size:
            resized_image = _pad_bottom_right(resized_image, self.target_size, self.target_size, self.pad_rgb)
        im_info = torch.tensor([float(resized_h), float(resized_w), float(resize_scale)], dtype=torch.float32)
        return resized_image, transformed_boxes, im_info

    def __getitem__(self, index):
        sample_index = int(index)
        last_reason = 'unknown'
        for _sample_attempt in range(3):
            frame_name, frame_annotation = self.samples[sample_index]
            frame_path = os.path.join(self.frames_path, frame_name)
            image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if image is None:
                last_reason = 'failed to read frame {}'.format(frame_path)
                sample_index = random.randrange(len(self.samples))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            boxes = self._annotation_to_boxes(frame_annotation)
            if boxes.shape[0] == 0:
                last_reason = 'frame {} has no valid GT boxes before transform'.format(frame_name)
                sample_index = random.randrange(len(self.samples))
                continue

            transformed_image = None
            transformed_boxes = None
            im_info = None
            retry_budget = self.max_crop_retries if self.mode == 'train' else 1
            for _crop_attempt in range(retry_budget):
                candidate_image, candidate_boxes, candidate_im_info = self._transform_with_lsj(image, boxes)
                if candidate_boxes.shape[0] > 0:
                    transformed_image = candidate_image
                    transformed_boxes = candidate_boxes
                    im_info = candidate_im_info
                    break

            if transformed_boxes is None or transformed_boxes.shape[0] == 0:
                transformed_image, transformed_boxes, im_info = self._transform_keep_all(image, boxes)
            if transformed_boxes.shape[0] == 0:
                last_reason = 'frame {} lost all GT boxes after transform fallback'.format(frame_name)
                sample_index = random.randrange(len(self.samples))
                continue

            image_tensor = torch.from_numpy(np.ascontiguousarray(transformed_image)).permute(2, 0, 1)
            gt_boxes = torch.from_numpy(np.ascontiguousarray(transformed_boxes))
            return image_tensor, im_info, gt_boxes, frame_name

        raise RuntimeError('Failed to build a valid Phase3 detector sample after retries: {}'.format(last_reason))


def detector_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    im_infos = torch.stack([item[1] for item in batch], dim=0)

    max_boxes = max(1, max(item[2].shape[0] for item in batch))
    gt_boxes = torch.zeros((len(batch), max_boxes, 5), dtype=torch.float32)
    num_boxes = torch.zeros((len(batch),), dtype=torch.int64)
    frame_names = []

    for batch_idx, (_, _, boxes, frame_name) in enumerate(batch):
        if boxes.shape[0] > 0:
            gt_boxes[batch_idx, : boxes.shape[0]] = boxes
            num_boxes[batch_idx] = int(boxes.shape[0])
        frame_names.append(frame_name)

    return images, im_infos, gt_boxes, num_boxes, frame_names
