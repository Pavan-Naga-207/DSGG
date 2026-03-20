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
try:
    from phase2_vitdet.adapter import ViTDetBackbone
except ImportError:
    _special_topics_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if _special_topics_root not in sys.path:
        sys.path.insert(0, _special_topics_root)
    from phase2_vitdet.adapter import ViTDetBackbone

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
        if self.backbone_name not in ('resnet101', 'vitdet'):
            raise ValueError('Unsupported backbone: {}. Expected resnet101 or vitdet'.format(backbone))

        self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load('fasterRCNN/models/faster_rcnn_ag.pth', map_location='cpu')
        self.fasterRCNN.load_state_dict(checkpoint['model'])

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
                for i, j in enumerate(DETECTOR_FOUND_IDX):

                    for k, kk in enumerate(GT_RELATIONS[i]):
                        if 'person_bbox' in kk.keys():
                            kkk = k
                            break
                    localhuman = int(global_idx[FINAL_BBOXES_X[:, 0] == i][kkk].item())

                    for m, n in enumerate(j):
                        if 'class' in GT_RELATIONS[i][m].keys():
                            im_idx.append(i)

                            pair.append([localhuman, int(global_idx[FINAL_BBOXES_X[:, 0] == i][int(n)].item())])

                            a_rel.append(GT_RELATIONS[i][m]['attention_relationship'].tolist())
                            s_rel.append(GT_RELATIONS[i][m]['spatial_relationship'].tolist())
                            c_rel.append(GT_RELATIONS[i][m]['contacting_relationship'].tolist())

                pair = torch.tensor(pair, device=device)
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
                base_feat = self.fasterRCNN.RCNN_base(inputs_data)
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

