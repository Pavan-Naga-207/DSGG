"""
Let's get the relationships yo
"""

import numpy as np
import os
import torch
import torch.nn as nn

from lib.word_vectors import obj_edge_vectors
from lib.transformer import transformer
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes


class _ShapeAwareNorm(nn.Module):
    """
    Use LayerNorm for token/feature vectors, but fall back to GroupNorm(1, C)
    for 4D feature maps [B, C, H, W].
    """

    def __init__(self, channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)
        self.group_norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        if x.dim() == 4:
            return self.group_norm(x)
        return self.layer_norm(x)


class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet', obj_classes=None):
        super(ObjectClassifier, self).__init__()
        self.classes = obj_classes
        self.mode = mode

        #----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img =64
        self.thresh = 0.01

        self.sgcls_duplicate_policy = os.environ.get('SGCLS_DUPLICATE_POLICY', 'legacy').strip().lower()
        policy_aliases = {
            'current': 'legacy',
            'legacy_mode': 'legacy',
            'off': 'none',
            'disabled': 'none',
            'iou_only': 'iou',
            'soft_iou': 'iou',
        }
        self.sgcls_duplicate_policy = policy_aliases.get(
            self.sgcls_duplicate_policy,
            self.sgcls_duplicate_policy,
        )
        if self.sgcls_duplicate_policy not in ('legacy', 'none', 'iou'):
            print(
                'warning: unknown SGCLS_DUPLICATE_POLICY={} -> using legacy'.format(
                    self.sgcls_duplicate_policy
                )
            )
            self.sgcls_duplicate_policy = 'legacy'
        self.sgcls_duplicate_iou = float(os.environ.get('SGCLS_DUPLICATE_IOU_THRESHOLD', '0.7'))
        self.sgcls_label_source = os.environ.get('SGCLS_LABEL_SOURCE', 'decoder').strip().lower()
        label_aliases = {
            'default': 'decoder',
            'sttran': 'decoder',
            'rcnn': 'detector',
            'det': 'detector',
            'input': 'detector',
        }
        self.sgcls_label_source = label_aliases.get(self.sgcls_label_source, self.sgcls_label_source)
        if self.sgcls_label_source not in ('decoder', 'detector'):
            print(
                'warning: unknown SGCLS_LABEL_SOURCE={} -> using decoder'.format(
                    self.sgcls_label_source
                )
            )
            self.sgcls_label_source = 'decoder'
        self.sgcls_debug_once = os.environ.get('SGCLS_DEBUG_ONCE', '0') == '1'
        self.sgcls_debug_always = os.environ.get('SGCLS_DEBUG_ALWAYS', '0') == '1'
        self.sgcls_debug_batch = int(os.environ.get('SGCLS_DEBUG_BATCH', '0'))
        self.sgcls_debug_topk = int(os.environ.get('SGCLS_DEBUG_TOPK', '20'))
        self.sgcls_debug_max_frames = int(os.environ.get('SGCLS_DEBUG_MAX_FRAMES', '6'))
        self._sgcls_eval_call_idx = 0
        self._sgcls_settings_printed = False

        #roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes)-1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(_ShapeAwareNorm(4),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 2048
        self.decoder_lin = nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                         _ShapeAwareNorm(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.classes)))

    def _print_sgcls_settings_once(self):
        if self.mode != 'sgcls' or self._sgcls_settings_printed:
            return
        self._sgcls_settings_printed = True
        print(
            'SGCLS duplicate policy: {} (iou_threshold={:.3f})  label_source={}'.format(
                self.sgcls_duplicate_policy,
                self.sgcls_duplicate_iou,
                self.sgcls_label_source,
            )
        )
        if self.sgcls_debug_once or self.sgcls_debug_always:
            print(
                'SGCLS debug enabled: once={} always={} batch={} topk={} max_frames={}'.format(
                    self.sgcls_debug_once,
                    self.sgcls_debug_always,
                    self.sgcls_debug_batch,
                    self.sgcls_debug_topk,
                    self.sgcls_debug_max_frames,
                )
            )

    def _should_debug_sgcls(self, call_idx):
        if self.mode != 'sgcls' or self.training:
            return False
        if self.sgcls_debug_always:
            return True
        if self.sgcls_debug_once and call_idx == self.sgcls_debug_batch:
            return True
        return False

    def _label_name(self, class_idx):
        class_idx = int(class_idx)
        if 0 <= class_idx < len(self.classes):
            return self.classes[class_idx]
        return 'unknown'

    def _class_histogram(self, labels):
        if labels.numel() == 0:
            return {}
        unique_labels, counts = torch.unique(labels, return_counts=True)
        histogram = {}
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            histogram[self._label_name(label)] = int(count)
        return histogram

    def _box_iou(self, box, boxes):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=box.dtype, device=box.device)
        x1 = torch.maximum(box[0], boxes[:, 0])
        y1 = torch.maximum(box[1], boxes[:, 1])
        x2 = torch.minimum(box[2], boxes[:, 2])
        y2 = torch.minimum(box[3], boxes[:, 3])
        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter = inter_w * inter_h
        area_box = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        union = area_box + area_boxes - inter
        return inter / union.clamp(min=1e-6)

    def _relabel_sgcls_duplicate(self, entry, global_idx, duplicate_class, stats, frame_idx, max_iou=None):
        duplicate_class = int(duplicate_class)
        if duplicate_class <= 0:
            return
        class_col = duplicate_class - 1
        if class_col >= entry['distribution'].shape[1]:
            return
        old_label = int(entry['pred_labels'][global_idx].item())
        old_name = self._label_name(old_label)
        entry['distribution'][global_idx, class_col] = 0
        new_score, new_label_offset = torch.max(entry['distribution'][global_idx], dim=0)
        new_label = int(new_label_offset.item()) + 1
        entry['pred_labels'][global_idx] = new_label
        entry['pred_scores'][global_idx] = new_score
        stats['relabeled'] += 1
        stats['frames_changed'][int(frame_idx)] = stats['frames_changed'].get(int(frame_idx), 0) + 1
        stats['changes_by_from'][old_name] = stats['changes_by_from'].get(old_name, 0) + 1
        stats['changes_by_to'][self._label_name(new_label)] = stats['changes_by_to'].get(self._label_name(new_label), 0) + 1
        if max_iou is not None:
            stats['max_iou_relabels'].append(float(max_iou))

    def _apply_sgcls_duplicate_policy(self, entry, box_idx, global_idx):
        before_labels = entry['pred_labels'].detach().clone()
        stats = {
            'policy': self.sgcls_duplicate_policy,
            'frames_changed': {},
            'changes_by_from': {},
            'changes_by_to': {},
            'max_iou_relabels': [],
            'relabeled': 0,
            'dropped_boxes': 0,
        }

        if self.sgcls_duplicate_policy == 'none':
            after_labels = entry['pred_labels'].detach().clone()
            return before_labels, after_labels, stats

        num_frames = int(box_idx[-1].item()) + 1 if box_idx.numel() > 0 else 0
        for frame_idx in range(num_frames):
            present = box_idx == frame_idx
            frame_indices = global_idx[present]
            if frame_indices.numel() <= 1:
                continue
            frame_labels = entry['pred_labels'][present]

            if self.sgcls_duplicate_policy == 'legacy':
                duplicate_class = torch.mode(frame_labels)[0]
                duplicate_mask = frame_labels == duplicate_class
                if torch.sum(duplicate_mask).item() <= 1:
                    continue
                duplicate_indices = frame_indices[duplicate_mask]
                duplicate_scores = entry['distribution'][duplicate_indices, int(duplicate_class.item()) - 1]
                order = torch.argsort(duplicate_scores)
                for order_idx in order[:-1]:
                    self._relabel_sgcls_duplicate(
                        entry=entry,
                        global_idx=duplicate_indices[order_idx],
                        duplicate_class=int(duplicate_class.item()),
                        stats=stats,
                        frame_idx=frame_idx,
                    )
                continue

            # Softer SGCLS ablation: only relabel same-class duplicates when
            # their GT boxes overlap heavily, which is closer to the SGDET NMS idea.
            for duplicate_class in torch.unique(frame_labels).tolist():
                duplicate_class = int(duplicate_class)
                if duplicate_class <= 0:
                    continue
                duplicate_mask = frame_labels == duplicate_class
                if torch.sum(duplicate_mask).item() <= 1:
                    continue
                duplicate_indices = frame_indices[duplicate_mask]
                duplicate_scores = entry['distribution'][duplicate_indices, duplicate_class - 1]
                order = torch.argsort(duplicate_scores, descending=True)
                kept = []
                for order_idx in order:
                    current_idx = duplicate_indices[order_idx]
                    if not kept:
                        kept.append(int(current_idx.item()))
                        continue
                    kept_tensor = torch.tensor(kept, dtype=torch.long, device=entry['boxes'].device)
                    current_box = entry['boxes'][current_idx, 1:]
                    kept_boxes = entry['boxes'][kept_tensor, 1:]
                    max_iou = float(self._box_iou(current_box, kept_boxes).max().item())
                    if max_iou >= self.sgcls_duplicate_iou:
                        self._relabel_sgcls_duplicate(
                            entry=entry,
                            global_idx=current_idx,
                            duplicate_class=duplicate_class,
                            stats=stats,
                            frame_idx=frame_idx,
                            max_iou=max_iou,
                        )
                    else:
                        kept.append(int(current_idx.item()))

        after_labels = entry['pred_labels'].detach().clone()
        return before_labels, after_labels, stats

    def _log_sgcls_debug(self, entry, box_idx, call_idx, before_labels, after_labels, stats):
        changed_mask = before_labels != after_labels
        changed_indices = torch.nonzero(changed_mask, as_tuple=False).view(-1)
        before_nonhuman = int(torch.sum(before_labels != 1).item())
        after_nonhuman = int(torch.sum(after_labels != 1).item())
        gt_labels = entry['labels']
        nonhuman_gt_mask = gt_labels != 1
        gt_match_before = float((before_labels == gt_labels).float().mean().item()) if gt_labels.numel() > 0 else 0.0
        gt_match_after = float((after_labels == gt_labels).float().mean().item()) if gt_labels.numel() > 0 else 0.0
        nonhuman_gt_match_before = (
            float((before_labels[nonhuman_gt_mask] == gt_labels[nonhuman_gt_mask]).float().mean().item())
            if torch.any(nonhuman_gt_mask)
            else 0.0
        )
        nonhuman_gt_match_after = (
            float((after_labels[nonhuman_gt_mask] == gt_labels[nonhuman_gt_mask]).float().mean().item())
            if torch.any(nonhuman_gt_mask)
            else 0.0
        )
        pair_count = int(entry['pair_idx'].shape[0]) if 'pair_idx' in entry else 0
        num_frames = int(box_idx[-1].item()) + 1 if box_idx.numel() > 0 else 0
        topk = min(self.sgcls_debug_topk, int(before_labels.numel()))

        print('\n' + '!' * 72)
        print('SGCLS DEBUG batch={} policy={} iou_threshold={:.3f} label_source={}'.format(
            call_idx,
            stats['policy'],
            self.sgcls_duplicate_iou,
            self.sgcls_label_source,
        ))
        print('gt_boxes_in: {}  frames: {}  pair_count_after: {}'.format(
            int(entry['labels'].numel()),
            num_frames,
            pair_count,
        ))
        print('nonhuman_boxes before/after: {}/{}'.format(before_nonhuman, after_nonhuman))
        print('relabeled_boxes: {}  dropped_boxes: {}'.format(
            stats['relabeled'],
            stats['dropped_boxes'],
        ))
        print(
            'gt_match before/after: {:.4f}/{:.4f}  nonhuman_gt_match before/after: {:.4f}/{:.4f}'.format(
                gt_match_before,
                gt_match_after,
                nonhuman_gt_match_before,
                nonhuman_gt_match_after,
            )
        )
        if stats['max_iou_relabels']:
            print(
                'iou-relabel max/mean: {:.4f}/{:.4f}'.format(
                    max(stats['max_iou_relabels']),
                    float(np.mean(stats['max_iou_relabels'])),
                )
            )

        common_classes = ['person', 'chair', 'table', 'cup/glass/bottle', 'phone/camera', 'laptop', 'sofa/couch']
        common_summary = []
        for class_name in common_classes:
            from_count = stats['changes_by_from'].get(class_name, 0)
            to_count = stats['changes_by_to'].get(class_name, 0)
            common_summary.append('{}:{}->{}'.format(class_name, from_count, to_count))
        print('common_class_changes (from->to counts): {}'.format(', '.join(common_summary)))

        print('before_histogram:', self._class_histogram(before_labels))
        print('after_histogram:', self._class_histogram(after_labels))

        if topk > 0:
            before_preview = before_labels[:topk].tolist()
            after_preview = after_labels[:topk].tolist()
            print('before_labels_first_{}: {}'.format(topk, before_preview))
            print('before_names_first_{}: {}'.format(topk, [self._label_name(label) for label in before_preview]))
            print('after_labels_first_{}: {}'.format(topk, after_preview))
            print('after_names_first_{}: {}'.format(topk, [self._label_name(label) for label in after_preview]))

        if changed_indices.numel() > 0:
            change_preview = changed_indices[:topk].tolist()
            print('changed_box_indices_first_{}: {}'.format(len(change_preview), change_preview))
            for idx in change_preview:
                print(
                    '  box {} frame {}: {} -> {}'.format(
                        idx,
                        int(box_idx[idx].item()),
                        self._label_name(before_labels[idx]),
                        self._label_name(after_labels[idx]),
                    )
                )
        else:
            print('no label changes from SGCLS duplicate policy on this batch.')

        max_frames = min(self.sgcls_debug_max_frames, num_frames)
        for frame_idx in range(max_frames):
            frame_mask = box_idx == frame_idx
            frame_before = before_labels[frame_mask].tolist()
            frame_after = after_labels[frame_mask].tolist()
            print(
                'frame {} labels before: {} | after: {}'.format(
                    frame_idx,
                    [self._label_name(label) for label in frame_before],
                    [self._label_name(label) for label in frame_after],
                )
            )

        print('!' * 72 + '\n')

    def clean_class(self, entry, b, class_idx):
        final_boxes = []
        final_dists = []
        final_feats = []
        final_labels = []
        for i in range(b):
            scores = entry['distribution'][entry['boxes'][:, 0] == i]
            pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i]
            feats = entry['features'][entry['boxes'][:, 0] == i]
            pred_labels = entry['pred_labels'][entry['boxes'][:, 0] == i]

            new_box = pred_boxes[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_feats = feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores = scores[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores[:, class_idx-1] = 0
            if new_scores.shape[0] > 0:
                new_labels = torch.argmax(new_scores, dim=1) + 1
            else:
                new_labels = torch.tensor([], dtype=torch.long).to(scores.device)

            final_dists.append(scores)
            final_dists.append(new_scores)
            final_boxes.append(pred_boxes)
            final_boxes.append(new_box)
            final_feats.append(feats)
            final_feats.append(new_feats)
            final_labels.append(pred_labels)
            final_labels.append(new_labels)

        entry['boxes'] = torch.cat(final_boxes, dim=0)
        entry['distribution'] = torch.cat(final_dists, dim=0)
        entry['features'] = torch.cat(final_feats, dim=0)
        entry['pred_labels'] = torch.cat(final_labels, dim=0)
        return entry

    def _set_empty_relations(self, entry, device):
        # Keep tensor shapes explicit so downstream evaluator logic can handle
        # "no relations in this frame" without shape/index errors.
        box_dtype = entry['boxes'].dtype if 'boxes' in entry else torch.float32
        entry['pair_idx'] = torch.empty((0, 2), dtype=torch.long, device=device)
        entry['im_idx'] = torch.empty((0,), dtype=torch.float, device=device)
        entry['union_feat'] = torch.empty((0, 1024, 7, 7), dtype=torch.float, device=device)
        entry['union_box'] = torch.empty((0, 5), dtype=box_dtype, device=device)
        entry['spatial_masks'] = torch.empty((0, 2, 27, 27), dtype=torch.float, device=device)
        if 'pred_labels' not in entry:
            if 'labels' in entry:
                entry['pred_labels'] = entry['labels']
            else:
                entry['pred_labels'] = torch.empty((0,), dtype=torch.long, device=device)
        if 'pred_scores' not in entry:
            if 'scores' in entry:
                entry['pred_scores'] = entry['scores']
            else:
                entry['pred_scores'] = torch.empty((0,), dtype=torch.float, device=device)
        return entry

    def forward(self, entry):
        self._print_sgcls_settings_once()

        if self.mode  == 'predcls':
            entry['pred_labels'] = entry['labels']
            return entry
        elif self.mode == 'sgcls':

            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            if self.training:
                entry['distribution'] = self.decoder_lin(obj_features)
                entry['pred_labels'] = entry['labels']
            else:
                call_idx = self._sgcls_eval_call_idx
                self._sgcls_eval_call_idx += 1
                detector_distribution = entry['distribution']
                if self.sgcls_label_source == 'decoder':
                    entry['distribution'] = self.decoder_lin(obj_features)
                    entry['distribution'] = torch.softmax(entry['distribution'][:, 1:], dim=1)
                else:
                    entry['distribution'] = detector_distribution

                box_idx = entry['boxes'][:,0].long()
                if box_idx.numel() == 0:
                    return self._set_empty_relations(entry, obj_features.device)
                b = int(box_idx[-1] + 1)

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0], device=obj_features.device)

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0]) # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                before_labels, after_labels, sgcls_stats = self._apply_sgcls_duplicate_policy(
                    entry=entry,
                    box_idx=box_idx,
                    global_idx=global_idx,
                )


                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx==j][entry['pred_labels'][box_idx==j] != 1]: # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                if len(pair) == 0:
                    return self._set_empty_relations(entry, obj_features.device)

                pair = torch.tensor(pair, dtype=torch.long).to(obj_features.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(obj_features.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx

                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat((im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                                        torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(obj_features.device)
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                entry['spatial_masks'] = spatial_masks
                if self._should_debug_sgcls(call_idx):
                    self._log_sgcls_debug(
                        entry=entry,
                        box_idx=box_idx,
                        call_idx=call_idx,
                        before_labels=before_labels,
                        after_labels=after_labels,
                        stats=sgcls_stats,
                    )
            return entry
        else:
            if self.training:
                obj_embed = entry['distribution'] @ self.obj_embed.weight
                pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
                obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)

                box_idx = entry['boxes'][:, 0][entry['pair_idx'].unique()]
                l = torch.sum(box_idx == torch.mode(box_idx)[0])
                b = int(box_idx[-1] + 1)  # !!!

                entry['distribution'] = self.decoder_lin(obj_features)
                entry['pred_labels'] = entry['labels']
            else:

                obj_embed = entry['distribution'] @ self.obj_embed.weight
                pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
                obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1) #use the result from FasterRCNN directly

                box_idx = entry['boxes'][:, 0].long()
                if box_idx.numel() == 0:
                    return self._set_empty_relations(entry, obj_features.device)
                b = int(box_idx[-1] + 1)

                entry = self.clean_class(entry, b, 5)
                entry = self.clean_class(entry, b, 8)
                entry = self.clean_class(entry, b, 17)

                # # NMS
                final_boxes = []
                final_dists = []
                final_feats = []
                for i in range(b):
                    # images in the batch
                    scores = entry['distribution'][entry['boxes'][:, 0] == i]
                    pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
                    feats = entry['features'][entry['boxes'][:, 0] == i]

                    for j in range(len(self.classes) - 1):
                        # NMS according to obj categories
                        inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
                        # if there is det
                        if inds.numel() > 0:
                            cls_dists = scores[inds]
                            cls_feats = feats[inds]
                            cls_scores = cls_dists[:, j]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds]
                            cls_dists = cls_dists[order]
                            cls_feats = cls_feats[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)  # hyperparameter

                            final_dists.append(cls_dists[keep.view(-1).long()])
                            final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
                                                                                                        1).to(cls_boxes.device),
                                                          cls_boxes[order, :][keep.view(-1).long()]), 1))
                            final_feats.append(cls_feats[keep.view(-1).long()])

                if len(final_boxes) == 0:
                    return self._set_empty_relations(entry, entry['boxes'].device)

                entry['boxes'] = torch.cat(final_boxes, dim=0)
                box_idx = entry['boxes'][:, 0].long()
                entry['distribution'] = torch.cat(final_dists, dim=0)
                entry['features'] = torch.cat(final_feats, dim=0)

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.full([b, 1], fill_value=-1, dtype=torch.int64, device=box_idx.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0], device=box_idx.device)

                for i in range(b):
                    frame_mask = box_idx == i
                    if torch.any(frame_mask):
                        # Local bbox index with highest human score in this frame.
                        local_human_idx = torch.argmax(entry['distribution'][frame_mask, 0])
                        HUMAN_IDX[i] = global_idx[frame_mask][local_human_idx]

                valid_human_idx = HUMAN_IDX.squeeze(1) >= 0
                if torch.any(valid_human_idx):
                    chosen_humans = HUMAN_IDX.squeeze(1)[valid_human_idx]
                    entry['pred_labels'][chosen_humans] = 1
                    entry['pred_scores'][chosen_humans] = entry['distribution'][chosen_humans, 0]

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    if i.item() < 0:
                        continue
                    for m in global_idx[box_idx == j][
                        entry['pred_labels'][box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                if len(pair) == 0:
                    return self._set_empty_relations(entry, box_idx.device)

                pair = torch.tensor(pair, dtype=torch.long).to(box_idx.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx
                entry['human_idx'] = HUMAN_IDX
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat(
                    (im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                     torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)

            return entry


class STTran(nn.Module):

    def __init__(self, mode='sgdet',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None, rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(STTran, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode

        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)

        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, 256 // 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, 256),
        )
        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)
        self.vr_fc = nn.Linear(256*7*7, 512)

        wv_cache_dir = os.environ.get('STTRAN_WV_DIR', 'data')
        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir=wv_cache_dir, wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter')

        self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(1936, self.contact_class_num)

    def forward(self, entry):

        entry = self.object_classifier(entry)
        if ('pair_idx' not in entry) or entry['pair_idx'].numel() == 0:
            device = entry['boxes'].device
            entry["attention_distribution"] = torch.empty((0, self.attention_class_num), device=device)
            entry["spatial_distribution"] = torch.empty((0, self.spatial_class_num), device=device)
            entry["contacting_distribution"] = torch.empty((0, self.contact_class_num), device=device)
            return entry

        # visual part
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

        # semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)
        # Spatial-Temporal Transformer
        global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'])

        entry["attention_distribution"] = self.a_rel_compress(global_output)
        entry["spatial_distribution"] = self.s_rel_compress(global_output)
        entry["contacting_distribution"] = self.c_rel_compress(global_output)

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])

        return entry
