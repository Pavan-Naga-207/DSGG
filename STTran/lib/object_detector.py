import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import cv2
import os
import sys

from lib.funcs import assign_relations
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fasterRCNN.lib.model.roi_layers import nms
from fasterRCNN.lib.model.utils.net_utils import _smooth_l1_loss
try:
    from phase2_vitdet.adapter import ViTDetBackbone
except ImportError:
    _special_topics_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if _special_topics_root not in sys.path:
        sys.path.insert(0, _special_topics_root)
    from phase2_vitdet.adapter import ViTDetBackbone


def _valid_group_count(num_channels, preferred_groups):
    groups = min(max(1, int(preferred_groups)), int(num_channels))
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


def _replace_batchnorm_with_groupnorm(module, preferred_groups=32):
    replaced = 0
    for child_name, child in list(module.named_children()):
        replaced += _replace_batchnorm_with_groupnorm(child, preferred_groups=preferred_groups)
        if isinstance(child, nn.BatchNorm2d):
            num_groups = _valid_group_count(child.num_features, preferred_groups)
            gn = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=child.num_features,
                eps=child.eps,
                affine=child.affine,
            )
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, child_name, gn)
            replaced += 1
    return replaced


def _freeze_batchnorm_stats(module):
    frozen = 0
    for child in module.modules():
        if isinstance(child, nn.BatchNorm2d):
            child.eval()
            if child.affine:
                child.weight.requires_grad_(False)
                child.bias.requires_grad_(False)
            frozen += 1
    return frozen


def _reset_linear_head(linear_layer, init_std):
    if linear_layer is None:
        return
    nn.init.normal_(linear_layer.weight, mean=0.0, std=init_std)
    if linear_layer.bias is not None:
        nn.init.constant_(linear_layer.bias, 0.0)


def _reset_detector_prediction_heads(detector_module):
    _reset_linear_head(detector_module.RCNN_cls_score, init_std=0.01)
    _reset_linear_head(detector_module.RCNN_bbox_pred, init_std=0.001)


def _box_area(boxes):
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    return widths * heights


def _aligned_giou_loss(pred_boxes, target_boxes):
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0

    inter_x1 = torch.maximum(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.maximum(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.minimum(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.minimum(pred_boxes[:, 3], target_boxes[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    pred_area = _box_area(pred_boxes)
    target_area = _box_area(target_boxes)
    union = pred_area + target_area - inter_area
    iou = inter_area / union.clamp(min=torch.finfo(pred_boxes.dtype).eps)

    enc_x1 = torch.minimum(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.minimum(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.maximum(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.maximum(pred_boxes[:, 3], target_boxes[:, 3])
    enc_area = ((enc_x2 - enc_x1).clamp(min=0.0) * (enc_y2 - enc_y1).clamp(min=0.0))
    giou = iou - ((enc_area - union) / enc_area.clamp(min=torch.finfo(pred_boxes.dtype).eps))
    return (1.0 - giou).mean()


class detector(nn.Module):

    '''first part: object detection (image/video)'''

    def __init__(
        self,
        train,
        object_classes,
        use_SUPPLY,
        mode='predcls',
        backbone='resnet101',
        det_threshold=0.1,
    ):
        super(detector, self).__init__()

        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode
        self.backbone_name = backbone.lower()
        if det_threshold is None:
            det_threshold = float(os.environ.get('DET_THRESHOLD', '0.1'))
        self.det_threshold = float(det_threshold)
        if self.det_threshold < 0.0:
            raise ValueError('det_threshold must be non-negative, got {}'.format(self.det_threshold))
        self.max_train_rois = int(os.environ.get('VITDET_MAX_TRAIN_ROIS', '256'))
        self.max_eval_rois = int(os.environ.get('VITDET_MAX_EVAL_ROIS', '1000'))
        self.detector_chunk = max(1, int(os.environ.get('VITDET_DET_CHUNK', '10')))
        self.detector_cls_loss = os.environ.get('DETECTOR_CLS_LOSS', 'ce').strip().lower()
        if self.detector_cls_loss not in ('ce', 'weighted_ce'):
            raise ValueError(
                'Unsupported DETECTOR_CLS_LOSS: {} (expected ce or weighted_ce)'.format(
                    self.detector_cls_loss
                )
            )
        self.detector_bbox_loss = os.environ.get('DETECTOR_BBOX_LOSS', 'smoothl1').strip().lower()
        if self.detector_bbox_loss not in ('smoothl1', 'smooth_l1', 'giou'):
            raise ValueError(
                'Unsupported DETECTOR_BBOX_LOSS: {} (expected smoothl1 or giou)'.format(
                    self.detector_bbox_loss
                )
            )
        self.rcnn_cls_weights = None
        if self.backbone_name not in ('resnet101', 'vitdet'):
            raise ValueError('Unsupported backbone: {}. Expected resnet101 or vitdet'.format(backbone))

        self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        load_ag_ckpt = os.environ.get('DETECTOR_LOAD_AG_CKPT', '1') == '1'
        skip_ag_head_load = os.environ.get('DETECTOR_SKIP_AG_HEAD_LOAD', '0') == '1'
        reinit_roi_heads = os.environ.get('DETECTOR_REINIT_ROI_HEADS', '0') == '1'
        if load_ag_ckpt:
            checkpoint = torch.load('fasterRCNN/models/faster_rcnn_ag.pth', map_location='cpu')
            detector_state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            if skip_ag_head_load:
                head_prefixes = ('RCNN_cls_score.', 'RCNN_bbox_pred.')
                original_key_count = len(detector_state)
                detector_state = {
                    key: value
                    for key, value in detector_state.items()
                    if not any(key.startswith(prefix) for prefix in head_prefixes)
                }
                skipped_keys = original_key_count - len(detector_state)
                print('detector init: skipped AG detector head keys:', skipped_keys)
            missing_keys, unexpected_keys = self.fasterRCNN.load_state_dict(detector_state, strict=False)
            print(
                'detector init: AG checkpoint load missing/unexpected keys: {}/{}'.format(
                    len(missing_keys),
                    len(unexpected_keys),
                )
            )
        else:
            print('detector init: DETECTOR_LOAD_AG_CKPT=0; skipping AG detector checkpoint load.')
        if reinit_roi_heads:
            _reset_detector_prediction_heads(self.fasterRCNN)
            print('detector init: reinitialized RCNN_cls_score and RCNN_bbox_pred heads.')
        detector_bn_mode = os.environ.get('DETECTOR_BN_MODE', 'batchnorm').strip().lower()
        detector_gn_groups = int(os.environ.get('DETECTOR_GN_GROUPS', '32'))
        if detector_bn_mode == 'groupnorm':
            replaced_bn = _replace_batchnorm_with_groupnorm(
                self.fasterRCNN, preferred_groups=detector_gn_groups
            )
            print(
                'detector norm mode: groupnorm (groups={}); replaced BatchNorm2d layers: {}'.format(
                    detector_gn_groups, replaced_bn
                )
            )
        elif detector_bn_mode in ('frozen', 'frozenbn', 'frozen_batchnorm'):
            frozen_bn = _freeze_batchnorm_stats(self.fasterRCNN)
            print('detector norm mode: frozen batchnorm; frozen BatchNorm2d layers: {}'.format(frozen_bn))
        elif detector_bn_mode not in ('batchnorm', 'bn'):
            raise ValueError(
                'Unsupported DETECTOR_BN_MODE: {} (expected batchnorm, groupnorm, or frozen)'.format(
                    detector_bn_mode
                )
            )

        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

        self.vitdet = None
        if self.backbone_name == 'vitdet':
            # Use 1/16 stride map with 1024 channels to match RPN expectations.
            self.vitdet = ViTDetBackbone(
                model_name=os.environ.get('VITDET_MODEL', 'vit_base_patch16_224'),
                out_channels=int(os.environ.get('VITDET_OUT_CHANNELS', '256')),
                align_channels=int(os.environ.get('VITDET_ALIGN_CHANNELS', '1024')),
                freeze_backbone=os.environ.get('VITDET_FREEZE', '1') == '1',
            )
        print(
            'detector pretrain losses: cls={} bbox={}'.format(
                self.detector_cls_loss, self.detector_bbox_loss
            )
        )

    def set_rcnn_class_weights(self, class_weights):
        if class_weights is None:
            self.rcnn_cls_weights = None
            return
        if not torch.is_tensor(class_weights):
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.rcnn_cls_weights = class_weights.detach().float()

    def _extract_base_features(self, images):
        if self.backbone_name == 'vitdet':
            feat_dict = self.vitdet(images)
            return feat_dict['base']
        return self.fasterRCNN.RCNN_base(images)

    def _ensure_scalar_tensor(self, value, device, dtype=torch.float32):
        if torch.is_tensor(value):
            return value
        return torch.tensor(float(value), device=device, dtype=dtype)

    def _detector_train_chunk(self, im_data, im_info, gt_boxes, num_boxes):
        device = im_data.device
        if int(num_boxes.sum().item()) <= 0:
            return None
        base_feat = self._extract_base_features(im_data)

        rois, rpn_loss_cls, rpn_loss_bbox = self.fasterRCNN.RCNN_rpn(
            base_feat,
            im_info,
            gt_boxes,
            num_boxes,
        )
        try:
            roi_data = self.fasterRCNN.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        except ValueError as exc:
            if 'bg_num_rois = 0 and fg_num_rois = 0' in str(exc):
                return None
            raise
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

        rois_label = rois_label.view(-1).long()
        rois_target = rois_target.view(-1, rois_target.size(2))
        rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
        rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))

        rois_flat = rois.view(-1, 5)
        pooled_feat = self.fasterRCNN.RCNN_roi_align(base_feat, rois_flat)
        pooled_feat = self.fasterRCNN._head_to_tail(pooled_feat)

        bbox_pred = self.fasterRCNN.RCNN_bbox_pred(pooled_feat)
        if not self.fasterRCNN.class_agnostic:
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0),
                int(bbox_pred.size(1) / 4),
                4,
            )
            bbox_pred = torch.gather(
                bbox_pred_view,
                1,
                rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4),
            ).squeeze(1)

        cls_score = self.fasterRCNN.RCNN_cls_score(pooled_feat)
        cls_loss_weights = None
        if self.detector_cls_loss == 'weighted_ce' and self.rcnn_cls_weights is not None:
            cls_loss_weights = self.rcnn_cls_weights.to(device=cls_score.device, dtype=cls_score.dtype)
        rcnn_loss_cls = F.cross_entropy(cls_score, rois_label, weight=cls_loss_weights)

        if self.detector_bbox_loss in ('smoothl1', 'smooth_l1'):
            rcnn_loss_bbox = _smooth_l1_loss(
                bbox_pred,
                rois_target,
                rois_inside_ws,
                rois_outside_ws,
            )
        else:
            positive = rois_label > 0
            if torch.any(positive):
                delta_std = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=bbox_pred.dtype, device=device)
                delta_mean = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=bbox_pred.dtype, device=device)
                pred_deltas = bbox_pred * delta_std + delta_mean
                target_deltas = rois_target * delta_std + delta_mean
                proposal_boxes = rois_flat[:, 1:5]
                pred_boxes = bbox_transform_inv(
                    proposal_boxes.unsqueeze(0),
                    pred_deltas.unsqueeze(0),
                    1,
                ).squeeze(0)
                target_boxes = bbox_transform_inv(
                    proposal_boxes.unsqueeze(0),
                    target_deltas.unsqueeze(0),
                    1,
                ).squeeze(0)
                rcnn_loss_bbox = _aligned_giou_loss(pred_boxes[positive], target_boxes[positive])
            else:
                rcnn_loss_bbox = bbox_pred.sum() * 0.0

        rpn_loss_cls = self._ensure_scalar_tensor(rpn_loss_cls, device=device, dtype=rcnn_loss_cls.dtype)
        rpn_loss_bbox = self._ensure_scalar_tensor(rpn_loss_bbox, device=device, dtype=rcnn_loss_bbox.dtype)

        topk = min(5, cls_score.shape[1])
        top1_hits = int((cls_score.argmax(dim=1) == rois_label).sum().item())
        top5_hits = int(
            cls_score.topk(topk, dim=1).indices.eq(rois_label.unsqueeze(1)).any(dim=1).sum().item()
        )

        return {
            'rpn_loss_cls': rpn_loss_cls,
            'rpn_loss_bbox': rpn_loss_bbox,
            'rcnn_loss_cls': rcnn_loss_cls,
            'rcnn_loss_bbox': rcnn_loss_bbox,
            'roi_top1_hits': top1_hits,
            'roi_top5_hits': top5_hits,
            'roi_count': int(rois_label.numel()),
            'frame_count': int(im_data.shape[0]),
        }

    def forward_detector_pretrain(self, im_data, im_info, gt_boxes, num_boxes):
        device = im_data.device
        counter = 0
        loss_names = ('rpn_loss_cls', 'rpn_loss_bbox', 'rcnn_loss_cls', 'rcnn_loss_bbox')
        aggregated_losses = {
            name: torch.tensor(0.0, device=device, dtype=im_data.dtype if im_data.dtype.is_floating_point else torch.float32)
            for name in loss_names
        }
        total_frames = 0
        total_roi_count = 0
        total_top1_hits = 0
        total_top5_hits = 0
        skipped_chunks = 0

        while counter < im_data.shape[0]:
            end = min(counter + self.detector_chunk, im_data.shape[0])
            chunk_stats = self._detector_train_chunk(
                im_data[counter:end],
                im_info[counter:end],
                gt_boxes[counter:end],
                num_boxes[counter:end],
            )
            if chunk_stats is None:
                skipped_chunks += 1
                counter = end
                continue
            chunk_frames = max(1, int(chunk_stats['frame_count']))
            total_frames += chunk_frames
            total_roi_count += int(chunk_stats['roi_count'])
            total_top1_hits += int(chunk_stats['roi_top1_hits'])
            total_top5_hits += int(chunk_stats['roi_top5_hits'])
            for name in loss_names:
                aggregated_losses[name] = aggregated_losses[name] + chunk_stats[name] * chunk_frames
            counter = end

        if total_frames == 0:
            return None
        normalizer = float(max(1, total_frames))
        for name in loss_names:
            aggregated_losses[name] = aggregated_losses[name] / normalizer
        aggregated_losses['loss'] = sum(aggregated_losses[name] for name in loss_names)
        aggregated_losses['roi_top1'] = float(total_top1_hits) / float(max(1, total_roi_count))
        aggregated_losses['roi_top5'] = float(total_top5_hits) / float(max(1, total_roi_count))
        aggregated_losses['roi_count'] = total_roi_count
        aggregated_losses['frame_count'] = total_frames
        aggregated_losses['skipped_chunks'] = skipped_chunks
        return aggregated_losses

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):
        device = im_data.device

        if self.mode == 'sgdet':
            counter = 0
            counter_image = 0

            # create saved-bbox, labels, scores, features
            FINAL_BBOXES = torch.empty((0, 5), device=device)
            FINAL_LABELS = torch.empty(0, dtype=torch.int64, device=device)
            FINAL_SCORES = torch.empty(0, device=device)
            feat_dim = self.fasterRCNN.RCNN_cls_score.in_features
            FINAL_FEATURES = torch.empty((0, feat_dim), device=device)
            FINAL_BASE_FEATURES = None

            while counter < im_data.shape[0]:
                # compute a small frame chunk and collect all frames in the video.
                if counter + self.detector_chunk < im_data.shape[0]:
                    inputs_data = im_data[counter:counter + self.detector_chunk]
                    inputs_info = im_info[counter:counter + self.detector_chunk]
                    inputs_gtboxes = gt_boxes[counter:counter + self.detector_chunk]
                    inputs_numboxes = num_boxes[counter:counter + self.detector_chunk]

                else:
                    inputs_data = im_data[counter:]
                    inputs_info = im_info[counter:]
                    inputs_gtboxes = gt_boxes[counter:]
                    inputs_numboxes = num_boxes[counter:]

                if self.backbone_name == 'vitdet':
                    feat_dict = self.vitdet(inputs_data)
                    base_feat = feat_dict['base']
                    rois, rpn_loss_cls, rpn_loss_bbox = self.fasterRCNN.RCNN_rpn(
                        base_feat, inputs_info, inputs_gtboxes, inputs_numboxes
                    )
                    max_rois = self.max_train_rois if self.is_train else self.max_eval_rois
                    if max_rois > 0 and rois.shape[1] > max_rois:
                        rois = rois[:, :max_rois, :]
                    # Break potential shared storage from RPN internals before RoIAlign
                    # so autograd-saved proposal tensors are not modified in-place later.
                    rois = rois.clone()
                    use_proposal_target = self.training and torch.sum(inputs_numboxes).item() > 0
                    if use_proposal_target:
                        roi_data = self.fasterRCNN.RCNN_proposal_target(rois, inputs_gtboxes, inputs_numboxes)
                        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
                        rois_label = rois_label.view(-1).long()
                        rois_target = rois_target.view(-1, rois_target.size(2))
                        rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
                        rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))
                    else:
                        rois_label = None
                        rois_target = None
                        rois_inside_ws = None
                        rois_outside_ws = None
                        rpn_loss_cls = 0
                        rpn_loss_bbox = 0

                    pooled_feat = self.fasterRCNN.RCNN_roi_align(base_feat, rois.reshape(-1, 5).clone())
                    pooled_feat = self.fasterRCNN._head_to_tail(pooled_feat)
                    cls_score = self.fasterRCNN.RCNN_cls_score(pooled_feat)
                    cls_prob = torch.softmax(cls_score, 1)
                    bbox_pred = self.fasterRCNN.RCNN_bbox_pred(pooled_feat)
                    cls_prob = cls_prob.view(inputs_data.size(0), rois.size(1), -1)
                    bbox_pred = bbox_pred.view(inputs_data.size(0), rois.size(1), -1)
                    roi_features = pooled_feat.view(inputs_data.size(0), rois.size(1), -1)
                else:
                    rois, cls_prob, bbox_pred, base_feat, roi_features = self.fasterRCNN(inputs_data, inputs_info,
                                                                                         inputs_gtboxes, inputs_numboxes)

                SCORES = cls_prob.data
                boxes = rois.data[:, :, 1:5]
                # bbox regression (class specific)
                box_deltas = bbox_pred.data
                delta_std = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=box_deltas.dtype, device=device)
                delta_mean = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=box_deltas.dtype, device=device)
                box_deltas = box_deltas.view(-1, 4) * delta_std + delta_mean  # the first is normalize std, the second is mean
                box_deltas = box_deltas.view(-1, rois.shape[1], 4 * len(self.object_classes))  # post_NMS_NTOP: 30
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                PRED_BOXES = clip_boxes(pred_boxes, im_info.data, 1)

                PRED_BOXES /= inputs_info[0, 2] # original bbox scale!!!!!!!!!!!!!!

                #traverse frames
                for i in range(rois.shape[0]):
                    # images in the batch
                    scores = SCORES[i]
                    pred_boxes = PRED_BOXES[i]

                    for j in range(1, len(self.object_classes)):
                        # NMS according to obj categories
                        inds = torch.nonzero(scores[:, j] > self.det_threshold).view(-1)
                        # if there is det
                        if inds.numel() > 0:
                            cls_scores = scores[:, j][inds]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                            cls_dets = cls_dets[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.4) # NMS threshold
                            cls_dets = cls_dets[keep.view(-1).long()]

                            if j == 1:
                                # for person we only keep the highest score for person!
                                final_bbox = cls_dets[0,0:4].unsqueeze(0)
                                final_score = cls_dets[0,4].unsqueeze(0)
                                final_labels = torch.tensor([j], device=device)
                                final_features = roi_features[i, inds[order[keep][0]]].unsqueeze(0)
                            else:
                                final_bbox = cls_dets[:, 0:4]
                                final_score = cls_dets[:, 4]
                                final_labels = torch.tensor([j], device=device).repeat(keep.shape[0])
                                final_features = roi_features[i, inds[order[keep]]]

                            final_bbox = torch.cat((torch.tensor([[counter_image]], dtype=torch.float, device=device).repeat(final_bbox.shape[0], 1),
                                                    final_bbox), 1)
                            FINAL_BBOXES = torch.cat((FINAL_BBOXES, final_bbox), 0)
                            FINAL_LABELS = torch.cat((FINAL_LABELS, final_labels), 0)
                            FINAL_SCORES = torch.cat((FINAL_SCORES, final_score), 0)
                            FINAL_FEATURES = torch.cat((FINAL_FEATURES, final_features), 0)
                    if FINAL_BASE_FEATURES is None:
                        FINAL_BASE_FEATURES = base_feat[i].unsqueeze(0)
                    else:
                        FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat[i].unsqueeze(0)), 0)

                    counter_image += 1

                counter += self.detector_chunk
            FINAL_BBOXES = torch.clamp(FINAL_BBOXES, 0)
            if FINAL_BASE_FEATURES is None:
                FINAL_BASE_FEATURES = torch.empty((0,), device=device)
            prediction = {'FINAL_BBOXES': FINAL_BBOXES, 'FINAL_LABELS': FINAL_LABELS, 'FINAL_SCORES': FINAL_SCORES,
                          'FINAL_FEATURES': FINAL_FEATURES, 'FINAL_BASE_FEATURES': FINAL_BASE_FEATURES}

            if self.is_train:

                DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations(prediction, gt_annotation, assign_IOU_threshold=0.5)

                if self.use_SUPPLY:
                    # supply the unfounded gt boxes by detector into the scene graph generation training
                    FINAL_BBOXES_X = torch.empty((0, 5), device=device)
                    FINAL_LABELS_X = torch.empty(0, dtype=torch.int64, device=device)
                    FINAL_SCORES_X = torch.empty(0, device=device)
                    FINAL_FEATURES_X = torch.empty((0, feat_dim), device=device)
                    assigned_labels = torch.tensor(assigned_labels, dtype=torch.long).to(FINAL_BBOXES_X.device)

                    for i, j in enumerate(SUPPLY_RELATIONS):
                        if len(j) > 0:
                            unfound_gt_bboxes = torch.zeros([len(j), 5], device=device)
                            unfound_gt_classes = torch.zeros([len(j)], dtype=torch.int64, device=device)
                            one_scores = torch.ones([len(j)], dtype=torch.float32, device=device)  # probability
                            for m, n in enumerate(j):
                                # if person box is missing or objects
                                if 'bbox' in n.keys():
                                    bbox = torch.as_tensor(
                                        n['bbox'],
                                        dtype=unfound_gt_bboxes.dtype,
                                        device=unfound_gt_bboxes.device,
                                    )
                                    unfound_gt_bboxes[m, 1:] = bbox * im_info[i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = n['class']
                                else:
                                    # here happens always that IOU <0.5 but not unfounded
                                    person_bbox = torch.as_tensor(
                                        n['person_bbox'],
                                        dtype=unfound_gt_bboxes.dtype,
                                        device=unfound_gt_bboxes.device,
                                    )
                                    unfound_gt_bboxes[m, 1:] = person_bbox * im_info[i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = 1  # person class index

                            DETECTOR_FOUND_IDX[i] = list(np.concatenate((DETECTOR_FOUND_IDX[i],
                                                                         np.arange(
                                                                             start=int(sum(FINAL_BBOXES[:, 0] == i)),
                                                                             stop=int(
                                                                                 sum(FINAL_BBOXES[:, 0] == i)) + len(
                                                                                 SUPPLY_RELATIONS[i]))), axis=0).astype(
                                'int64'))

                            GT_RELATIONS[i].extend(SUPPLY_RELATIONS[i])

                            # compute the features of unfound gt_boxes
                            roi_boxes = unfound_gt_bboxes.clone()
                            pooled_feat = self.fasterRCNN.RCNN_roi_align(
                                FINAL_BASE_FEATURES[i].unsqueeze(0),
                                roi_boxes,
                            )
                            pooled_feat = self.fasterRCNN._head_to_tail(pooled_feat)
                            cls_prob = F.softmax(self.fasterRCNN.RCNN_cls_score(pooled_feat), 1)

                            unfound_gt_bboxes_norm = unfound_gt_bboxes.clone()
                            unfound_gt_bboxes_norm[:, 0] = i
                            unfound_gt_bboxes_norm[:, 1:] = unfound_gt_bboxes_norm[:, 1:] / im_info[i, 2]
                            FINAL_BBOXES_X = torch.cat(
                                (FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i], unfound_gt_bboxes_norm))
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i],
                                                        unfound_gt_classes))  # final label is not gt!
                            FINAL_SCORES_X = torch.cat(
                                (FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i], one_scores))
                            FINAL_FEATURES_X = torch.cat(
                                (FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i], pooled_feat))
                        else:
                            FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i]))
                            FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i]))

                FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES_X)[:, 1:], dim=1)
                global_idx = torch.arange(
                    start=0,
                    end=FINAL_BBOXES_X.shape[0],
                    device=FINAL_BBOXES_X.device,
                )  # all bbox indices

                im_idx = []  # which frame are the relations belong to
                pair = []
                a_rel = []
                s_rel = []
                c_rel = []
                for i, found_idx in enumerate(DETECTOR_FOUND_IDX):
                    frame_mask = FINAL_BBOXES_X[:, 0] == i
                    frame_global_idx = global_idx[frame_mask]
                    if frame_global_idx.numel() == 0:
                        continue

                    person_relation_idx = None
                    for rel_idx, rel in enumerate(GT_RELATIONS[i]):
                        if 'person_bbox' in rel.keys():
                            person_relation_idx = rel_idx
                            break

                    local_human_idx = None
                    if person_relation_idx is not None and person_relation_idx < len(found_idx):
                        local_human_idx = int(found_idx[person_relation_idx])
                    if local_human_idx is None:
                        frame_labels = FINAL_LABELS_X[frame_mask]
                        human_candidates = torch.nonzero(frame_labels == 1, as_tuple=False).view(-1)
                        if human_candidates.numel() > 0:
                            local_human_idx = int(human_candidates[0].item())
                        else:
                            local_human_idx = 0

                    local_human_idx = max(0, min(local_human_idx, int(frame_global_idx.numel()) - 1))
                    localhuman = int(frame_global_idx[local_human_idx].item())

                    for rel_idx, rel in enumerate(GT_RELATIONS[i]):
                        if 'class' not in rel.keys():
                            continue
                        if rel_idx >= len(found_idx):
                            continue
                        local_obj_idx = int(found_idx[rel_idx])
                        if local_obj_idx < 0 or local_obj_idx >= frame_global_idx.numel():
                            continue
                        obj_global_idx = int(frame_global_idx[local_obj_idx].item())
                        if obj_global_idx == localhuman:
                            continue

                        im_idx.append(i)
                        pair.append([localhuman, obj_global_idx])
                        a_rel.append(rel['attention_relationship'].tolist())
                        s_rel.append(rel['spatial_relationship'].tolist())
                        c_rel.append(rel['contacting_relationship'].tolist())

                if len(pair) == 0:
                    union_channels = (
                        int(FINAL_BASE_FEATURES.shape[1])
                        if FINAL_BASE_FEATURES.ndim == 4 and FINAL_BASE_FEATURES.shape[1] > 0
                        else 1024
                    )
                    entry = {'boxes': FINAL_BBOXES_X,
                             'labels': FINAL_LABELS_X,
                             'scores': FINAL_SCORES_X,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'im_idx': torch.empty((0,), dtype=torch.float, device=device),
                             'pair_idx': torch.empty((0, 2), dtype=torch.long, device=device),
                             'features': FINAL_FEATURES_X,
                             'union_feat': torch.empty(
                                 (0, union_channels, 7, 7),
                                 dtype=FINAL_BASE_FEATURES.dtype,
                                 device=device,
                             ),
                             'spatial_masks': torch.empty((0, 2, 27, 27), dtype=torch.float, device=device),
                             'attention_gt': a_rel,
                             'spatial_gt': s_rel,
                             'contacting_gt': c_rel}
                    return entry

                pair = torch.tensor(pair, dtype=torch.long, device=device)
                im_idx = torch.tensor(im_idx, dtype=torch.float, device=device)
                union_boxes = torch.cat((im_idx[:, None],
                                         torch.min(FINAL_BBOXES_X[:, 1:3][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES_X[:, 3:5][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 3:5][pair[:, 1]])), 1)

                union_boxes[:, 1:] = union_boxes[:, 1:] * im_info[0, 2]
                union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)

                pair_rois = torch.cat((FINAL_BBOXES_X[pair[:,0],1:],FINAL_BBOXES_X[pair[:,1],1:]), 1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES_X,
                         'labels': FINAL_LABELS_X,
                         'scores': FINAL_SCORES_X,
                         'distribution': FINAL_DISTRIBUTIONS,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'features': FINAL_FEATURES_X,
                         'union_feat': union_feat,
                         'spatial_masks': spatial_masks,
                         'attention_gt': a_rel,
                         'spatial_gt': s_rel,
                         'contacting_gt': c_rel}

                return entry

            else:
                FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                PRED_LABELS = PRED_LABELS + 1

                entry = {'boxes': FINAL_BBOXES,
                         'scores': FINAL_SCORES,
                         'distribution': FINAL_DISTRIBUTIONS,
                         'pred_labels': PRED_LABELS,
                         'features': FINAL_FEATURES,
                         'fmaps': FINAL_BASE_FEATURES,
                         'im_info': im_info[0, 2]}

                return entry
        else:
            # how many bboxes we have
            bbox_num = 0

            im_idx = []  # which frame are the relations belong to
            pair = []
            a_rel = []
            s_rel = []
            c_rel = []

            for i in gt_annotation:
                bbox_num += len(i)
            FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32, device=device)
            FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64, device=device)
            FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32, device=device)
            HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64, device=device)

            bbox_idx = 0
            for i, j in enumerate(gt_annotation):
                for m in j:
                    if 'person_bbox' in m.keys():
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0]).to(device=device)
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = 1
                        HUMAN_IDX[i] = bbox_idx
                        bbox_idx += 1
                    else:
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox']).to(device=device)
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = m['class']
                        im_idx.append(i)
                        pair.append([int(HUMAN_IDX[i]), bbox_idx])
                        a_rel.append(m['attention_relationship'].tolist())
                        s_rel.append(m['spatial_relationship'].tolist())
                        c_rel.append(m['contacting_relationship'].tolist())
                        bbox_idx += 1
            pair = torch.tensor(pair, device=device)
            im_idx = torch.tensor(im_idx, dtype=torch.float, device=device)

            counter = 0
            FINAL_BASE_FEATURES = torch.empty(0, device=device)

            while counter < im_data.shape[0]:
                #compute 10 images in batch and  collect all frames data in the video
                if counter + self.detector_chunk < im_data.shape[0]:
                    inputs_data = im_data[counter:counter + self.detector_chunk]
                else:
                    inputs_data = im_data[counter:]
                # Keep PredCLS/SGCLS on the same backbone path as SGDET so ViT
                # checkpoints do not silently fall back to the legacy ResNet base.
                base_feat = self._extract_base_features(inputs_data)
                FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat), 0)
                counter += self.detector_chunk

            FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]
            FINAL_FEATURES = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, FINAL_BBOXES)
            FINAL_FEATURES = self.fasterRCNN._head_to_tail(FINAL_FEATURES)

            if self.mode == 'predcls':

                union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
                FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES,
                         'labels': FINAL_LABELS, # here is the groundtruth
                         'scores': FINAL_SCORES,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'human_idx': HUMAN_IDX,
                         'features': FINAL_FEATURES,
                         'union_feat': union_feat,
                         'union_box': union_boxes,
                         'spatial_masks': spatial_masks,
                         'attention_gt': a_rel,
                         'spatial_gt': s_rel,
                         'contacting_gt': c_rel
                        }

                return entry
            elif self.mode == 'sgcls':
                if self.is_train:

                    FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    union_boxes = torch.cat(
                        (im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                    union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                    pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                          1).data.cpu().numpy()
                    spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                    entry = {'boxes': FINAL_BBOXES,
                             'labels': FINAL_LABELS,  # here is the groundtruth
                             'scores': FINAL_SCORES,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'pred_labels': PRED_LABELS,
                             'im_idx': im_idx,
                             'pair_idx': pair,
                             'human_idx': HUMAN_IDX,
                             'features': FINAL_FEATURES,
                             'union_feat': union_feat,
                             'union_box': union_boxes,
                             'spatial_masks': spatial_masks,
                             'attention_gt': a_rel,
                             'spatial_gt': s_rel,
                             'contacting_gt': c_rel}

                    return entry
                else:
                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]

                    FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    entry = {'boxes': FINAL_BBOXES,
                             'labels': FINAL_LABELS,  # here is the groundtruth
                             'scores': FINAL_SCORES,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'pred_labels': PRED_LABELS,
                             'im_idx': im_idx,
                             'pair_idx': pair,
                             'human_idx': HUMAN_IDX,
                             'features': FINAL_FEATURES,
                             'attention_gt': a_rel,
                             'spatial_gt': s_rel,
                             'contacting_gt': c_rel,
                             'fmaps': FINAL_BASE_FEATURES,
                             'im_info': im_info[0, 2]}

                    return entry
