import argparse
import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.sttran import STTran


def _build_loader(dataset: AG, num_workers: int, pin_memory: bool) -> DataLoader:
    kwargs = {
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": cuda_collate_fn,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(os.environ.get("PREFETCH_FACTOR", "2"))
    return DataLoader(dataset, **kwargs)


def _strip_ddp_prefix(state_dict):
    if state_dict is None:
        return None
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_checkpoint_state_dicts(path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device)
    model_state_dict = None
    detector_state_dict = None

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
        else:
            model_state_dict = checkpoint
        detector_state_dict = checkpoint.get("object_detector_state_dict")
    else:
        raise RuntimeError("Unsupported checkpoint format at {}".format(path))
    return _strip_ddp_prefix(model_state_dict), _strip_ddp_prefix(detector_state_dict)


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _safe_percentile(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _pct(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return 100.0 * float(numer) / float(denom)


def _frame_counts(frame_count: int, boxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        zeros = torch.zeros(frame_count, dtype=torch.long, device=boxes.device)
        return zeros, zeros, zeros

    frame_ids = boxes[:, 0].long()
    total = torch.bincount(frame_ids, minlength=frame_count)

    person_mask = labels == 1
    person = torch.bincount(frame_ids[person_mask], minlength=frame_count) if person_mask.any() else torch.zeros_like(total)

    object_mask = labels != 1
    objects = torch.bincount(frame_ids[object_mask], minlength=frame_count) if object_mask.any() else torch.zeros_like(total)
    return total, person, objects


@torch.no_grad()
def _collect_proposal_stats(
    object_detector_model: detector,
    im_data: torch.Tensor,
    im_info: torch.Tensor,
    gt_boxes: torch.Tensor,
    num_boxes: torch.Tensor,
    det_score_thresh: float,
) -> Dict[str, List[float]]:
    proposals_per_frame: List[float] = []
    proposal_mean_nonbg: List[float] = []
    proposal_max_nonbg: List[float] = []
    proposal_any_above_thresh: List[float] = []
    proposal_person_above_thresh: List[float] = []

    start = 0
    while start < im_data.shape[0]:
        end = min(start + 10, im_data.shape[0])
        inputs_data = im_data[start:end]
        inputs_info = im_info[start:end]
        inputs_gtboxes = gt_boxes[start:end]
        inputs_numboxes = num_boxes[start:end]

        if object_detector_model.backbone_name == "vitdet":
            feat_dict = object_detector_model.vitdet(inputs_data)
            base_feat = feat_dict["base"]
            rois, _, _ = object_detector_model.fasterRCNN.RCNN_rpn(base_feat, inputs_info, inputs_gtboxes, inputs_numboxes)
            pooled_feat = object_detector_model.fasterRCNN.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat = object_detector_model.fasterRCNN._head_to_tail(pooled_feat)
            cls_score = object_detector_model.fasterRCNN.RCNN_cls_score(pooled_feat)
            cls_prob = torch.softmax(cls_score, dim=1).view(inputs_data.size(0), rois.size(1), -1)
        else:
            rois, cls_prob, _, _, _ = object_detector_model.fasterRCNN(inputs_data, inputs_info, inputs_gtboxes, inputs_numboxes)

        max_nonbg = cls_prob[:, :, 1:].amax(dim=2)
        person_scores = cls_prob[:, :, 1]

        proposals_per_frame.extend([float(rois.shape[1])] * rois.shape[0])
        proposal_mean_nonbg.extend(max_nonbg.mean(dim=1).detach().cpu().tolist())
        proposal_max_nonbg.extend(max_nonbg.max(dim=1).values.detach().cpu().tolist())
        proposal_any_above_thresh.extend((max_nonbg > det_score_thresh).sum(dim=1).detach().cpu().tolist())
        proposal_person_above_thresh.extend((person_scores > det_score_thresh).sum(dim=1).detach().cpu().tolist())

        start = end

    return {
        "proposals_per_frame": proposals_per_frame,
        "proposal_mean_nonbg": proposal_mean_nonbg,
        "proposal_max_nonbg": proposal_max_nonbg,
        "proposal_any_above_thresh": proposal_any_above_thresh,
        "proposal_person_above_thresh": proposal_person_above_thresh,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SGDET collapse through proposal/detection/pair statistics.")
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--datasize", default="large", type=str)
    parser.add_argument("--backbone", default="vitdet", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-steps", default=25, type=int)
    parser.add_argument("--print-every", default=5, type=int)
    parser.add_argument("--det-score-thresh", default=0.1, type=float)
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(args.model_path))

    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    print("diagnostic_device:", device)
    print("model_path:", args.model_path)
    print("data_path:", args.data_path)
    print("backbone:", args.backbone)
    print("max_steps:", args.max_steps)
    print("det_score_thresh:", args.det_score_thresh)

    ag_test = AG(
        mode="test",
        datasize=args.datasize,
        data_path=args.data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=True,
    )
    dataloader = _build_loader(ag_test, args.num_workers, args.pin_memory)

    object_detector_model = detector(
        train=False,
        object_classes=ag_test.object_classes,
        use_SUPPLY=True,
        mode="sgdet",
        backbone=args.backbone,
        det_threshold=args.det_score_thresh,
    ).to(device=device)
    object_detector_model.eval()

    model = STTran(
        mode="sgdet",
        attention_class_num=len(ag_test.attention_relationships),
        spatial_class_num=len(ag_test.spatial_relationships),
        contact_class_num=len(ag_test.contacting_relationships),
        obj_classes=ag_test.object_classes,
        enc_layer_num=1,
        dec_layer_num=3,
    ).to(device=device)
    model.eval()

    model_state_dict, detector_state_dict = _load_checkpoint_state_dicts(args.model_path, device)
    if detector_state_dict is not None:
        det_missing_keys, det_unexpected_keys = object_detector_model.load_state_dict(detector_state_dict, strict=False)
        if det_missing_keys:
            print("detector_missing_keys:", len(det_missing_keys))
        if det_unexpected_keys:
            print("detector_unexpected_keys:", len(det_unexpected_keys))

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    print("checkpoint_missing_keys:", len(missing_keys))
    print("checkpoint_unexpected_keys:", len(unexpected_keys))

    videos_seen = 0
    frames_seen = 0
    total_boxes = 0
    total_person_boxes = 0
    total_object_boxes = 0
    score_values: List[float] = []

    frames_no_det = 0
    frames_only_person = 0
    frames_no_person = 0
    frames_with_objects = 0

    total_pairs = 0
    frames_no_pairs = 0
    videos_zero_pairs = 0

    proposals_per_frame: List[float] = []
    proposal_mean_nonbg: List[float] = []
    proposal_max_nonbg: List[float] = []
    proposal_any_above_thresh: List[float] = []
    proposal_person_above_thresh: List[float] = []

    with torch.no_grad():
        for b, data in enumerate(dataloader):
            if args.max_steps > 0 and b >= args.max_steps:
                break

            im_data = data[0].to(device=device, non_blocking=True)
            im_info = data[1].to(device=device, non_blocking=True)
            gt_boxes = data[2].to(device=device, non_blocking=True)
            num_boxes = data[3].to(device=device, non_blocking=True)
            gt_annotation = ag_test.gt_annotations[data[4]]

            prop_stats = _collect_proposal_stats(
                object_detector_model=object_detector_model,
                im_data=im_data,
                im_info=im_info,
                gt_boxes=gt_boxes,
                num_boxes=num_boxes,
                det_score_thresh=args.det_score_thresh,
            )
            proposals_per_frame.extend(prop_stats["proposals_per_frame"])
            proposal_mean_nonbg.extend(prop_stats["proposal_mean_nonbg"])
            proposal_max_nonbg.extend(prop_stats["proposal_max_nonbg"])
            proposal_any_above_thresh.extend(prop_stats["proposal_any_above_thresh"])
            proposal_person_above_thresh.extend(prop_stats["proposal_person_above_thresh"])

            det_entry = object_detector_model(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            boxes = det_entry["boxes"]
            labels = det_entry["pred_labels"]
            scores = det_entry["scores"]

            frame_count = int(im_data.shape[0])
            frame_total, frame_person, frame_objects = _frame_counts(frame_count, boxes, labels)

            frames_seen += frame_count
            videos_seen += 1
            total_boxes += int(boxes.shape[0])
            total_person_boxes += int((labels == 1).sum().item())
            total_object_boxes += int((labels != 1).sum().item())

            frames_no_det += int((frame_total == 0).sum().item())
            frames_only_person += int(((frame_total > 0) & (frame_objects == 0)).sum().item())
            frames_no_person += int((frame_person == 0).sum().item())
            frames_with_objects += int((frame_objects > 0).sum().item())

            if scores.numel() > 0:
                score_values.extend(scores.detach().cpu().tolist())

            cls_entry = model.object_classifier(copy.deepcopy(det_entry))
            if "pair_idx" in cls_entry and cls_entry["pair_idx"].numel() > 0:
                pair_im_idx = cls_entry["im_idx"].long()
                pair_per_frame = torch.bincount(pair_im_idx, minlength=frame_count)
            else:
                pair_per_frame = torch.zeros(frame_count, dtype=torch.long, device=device)

            pairs_this_video = int(pair_per_frame.sum().item())
            total_pairs += pairs_this_video
            frames_no_pairs += int((pair_per_frame == 0).sum().item())
            if pairs_this_video == 0:
                videos_zero_pairs += 1

            if args.print_every > 0 and ((b + 1) % args.print_every == 0):
                print(
                    "[step {:03d}] frames={} boxes={} person_boxes={} object_boxes={} pairs={} no_det_frames={} no_pair_frames={}".format(
                        b + 1,
                        frame_count,
                        int(boxes.shape[0]),
                        int((labels == 1).sum().item()),
                        int((labels != 1).sum().item()),
                        pairs_this_video,
                        int((frame_total == 0).sum().item()),
                        int((pair_per_frame == 0).sum().item()),
                    )
                )

    print("--------------------------------------------------")
    print("SGDET DIAGNOSTIC SUMMARY")
    print("videos_seen:", videos_seen)
    print("frames_seen:", frames_seen)

    print("avg_rpn_proposals_per_frame:", _safe_mean(proposals_per_frame))
    print("avg_proposals_any_score_above_{:.4f}:".format(args.det_score_thresh), _safe_mean(proposal_any_above_thresh))
    print("avg_proposals_person_score_above_{:.4f}:".format(args.det_score_thresh), _safe_mean(proposal_person_above_thresh))
    print("proposal_max_nonbg_p50:", _safe_median(proposal_max_nonbg))
    print("proposal_max_nonbg_p90:", _safe_percentile(proposal_max_nonbg, 90))
    print("proposal_mean_nonbg_p50:", _safe_median(proposal_mean_nonbg))

    print("avg_detected_boxes_per_frame:", (float(total_boxes) / float(frames_seen)) if frames_seen > 0 else 0.0)
    print("avg_detected_person_boxes_per_frame:", (float(total_person_boxes) / float(frames_seen)) if frames_seen > 0 else 0.0)
    print("avg_detected_object_boxes_per_frame:", (float(total_object_boxes) / float(frames_seen)) if frames_seen > 0 else 0.0)
    print("det_score_p10:", _safe_percentile(score_values, 10))
    print("det_score_p50:", _safe_median(score_values))
    print("det_score_p90:", _safe_percentile(score_values, 90))

    print("frames_no_det_pct:", _pct(frames_no_det, frames_seen))
    print("frames_only_person_pct:", _pct(frames_only_person, frames_seen))
    print("frames_no_person_pct:", _pct(frames_no_person, frames_seen))
    print("frames_with_objects_pct:", _pct(frames_with_objects, frames_seen))

    print("avg_pairs_per_frame:", (float(total_pairs) / float(frames_seen)) if frames_seen > 0 else 0.0)
    print("frames_no_pairs_pct:", _pct(frames_no_pairs, frames_seen))
    print("videos_zero_pairs_pct:", _pct(videos_zero_pairs, videos_seen))
    print("--------------------------------------------------")

    frames_no_det_pct = _pct(frames_no_det, frames_seen)
    frames_only_person_pct = _pct(frames_only_person, frames_seen)
    frames_no_pairs_pct = _pct(frames_no_pairs, frames_seen)
    mean_proposals_above_thresh = _safe_mean(proposal_any_above_thresh)

    if frames_no_det_pct > 90.0:
        if mean_proposals_above_thresh < 1.0:
            print("diagnosis: RPN/RCNN proposal scores are below the detector gate; confidence scale collapsed.")
        else:
            print("diagnosis: proposals exist but post-NMS detector gating is too strict for current score scale.")
    elif frames_only_person_pct > 70.0:
        print("diagnosis: detector keeps mostly person boxes; non-person object classification is collapsing.")
    elif frames_no_pairs_pct > 70.0:
        print("diagnosis: boxes exist but relation pair construction collapses in object classifier.")
    else:
        print("diagnosis: detector and pair construction both produce candidates; check relation head calibration next.")


if __name__ == "__main__":
    main()
