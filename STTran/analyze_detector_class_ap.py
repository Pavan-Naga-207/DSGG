import argparse
import json
import os

import numpy as np
import torch

from dataloader.action_genome import AG
from evaluate_detector_map import (
    _bbox_iou_single_to_many,
    _build_loader,
    _extract_gt_for_frame,
    _load_checkpoint_state_dicts,
    _load_state_dict_flexible,
    _voc_ap,
)
from lib.object_detector import detector


def build_gt_records(dataset, selected_videos):
    class_count = len(dataset.object_classes)
    gt_records = {cls: {} for cls in range(1, class_count)}
    gt_count = np.zeros(class_count, dtype=np.int64)

    for vid_idx in range(selected_videos):
        frame_names = dataset.video_list[vid_idx]
        gt_video = dataset.gt_annotations[vid_idx]
        num_frames = min(len(frame_names), len(gt_video))
        for frame_idx in range(num_frames):
            image_id = frame_names[frame_idx]
            class_to_boxes = _extract_gt_for_frame(gt_video[frame_idx])
            for cls, boxes in class_to_boxes.items():
                if cls <= 0 or cls >= class_count:
                    continue
                arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                if arr.shape[0] == 0:
                    continue
                gt_records[cls][image_id] = {
                    "boxes": arr,
                    "detected": np.zeros(arr.shape[0], dtype=np.bool_),
                }
                gt_count[cls] += arr.shape[0]
    return gt_records, gt_count


def evaluate_checkpoint(data_path, datasize, model_path, backbone, vit_model, det_threshold, num_workers, pin_memory, device, max_steps, max_video_frames, iou_threshold, use_07_metric):
    if vit_model:
        os.environ["VITDET_MODEL"] = vit_model

    dataset = AG(
        mode="test",
        datasize=datasize,
        data_path=data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=True,
        backbone=backbone,
    )
    dataloader = _build_loader(dataset, num_workers, pin_memory)
    class_count = len(dataset.object_classes)
    object_detector = detector(
        train=False,
        object_classes=dataset.object_classes,
        use_SUPPLY=True,
        mode="sgdet",
        backbone=backbone,
        det_threshold=det_threshold,
    ).to(device=device)
    object_detector.eval()

    _, detector_state_dict = _load_checkpoint_state_dicts(model_path, device)
    if detector_state_dict is not None:
        _load_state_dict_flexible(object_detector, detector_state_dict)

    selected_videos = len(dataset) if max_steps < 0 else min(len(dataset), int(max_steps))
    gt_records, gt_count = build_gt_records(dataset, selected_videos)
    detections = {cls: [] for cls in range(1, class_count)}

    with torch.no_grad():
        for b, data in enumerate(dataloader):
            if max_steps > 0 and b >= max_steps:
                break

            video_idx = int(data[4])
            frame_names = dataset.video_list[video_idx]
            gt_annotation = dataset.gt_annotations[video_idx]

            im_data = data[0].to(device=device, non_blocking=True)
            im_info = data[1].to(device=device, non_blocking=True)
            gt_boxes = data[2].to(device=device, non_blocking=True)
            num_boxes = data[3].to(device=device, non_blocking=True)

            if max_video_frames > 0 and im_data.shape[0] > max_video_frames:
                im_data = im_data[:max_video_frames]
                im_info = im_info[:max_video_frames]
                gt_boxes = gt_boxes[:max_video_frames]
                num_boxes = num_boxes[:max_video_frames]
                frame_names = frame_names[:max_video_frames]
                gt_annotation = gt_annotation[:max_video_frames]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            if not isinstance(entry, dict):
                continue
            if "boxes" not in entry or entry["boxes"].numel() == 0:
                continue

            boxes = entry["boxes"]
            labels = entry.get("pred_labels", entry.get("labels"))
            scores = entry.get("scores")
            if labels is None or scores is None:
                continue

            n = int(min(boxes.shape[0], labels.shape[0], scores.shape[0]))
            for i in range(n):
                cls = int(labels[i].item())
                if cls <= 0 or cls >= class_count:
                    continue
                frame_idx = int(boxes[i, 0].item())
                if frame_idx < 0 or frame_idx >= len(frame_names):
                    continue
                image_id = frame_names[frame_idx]
                bbox = boxes[i, 1:5].detach().cpu().float().numpy().astype(np.float32)
                score = float(scores[i].item())
                detections[cls].append((image_id, score, bbox))

    ap_per_class = np.full(class_count, np.nan, dtype=np.float64)
    recall_per_class = np.full(class_count, np.nan, dtype=np.float64)
    precision_per_class = np.full(class_count, np.nan, dtype=np.float64)

    for cls in range(1, class_count):
        npos = int(gt_count[cls])
        if npos == 0:
            continue

        cls_dets = detections[cls]
        if len(cls_dets) == 0:
            ap_per_class[cls] = 0.0
            recall_per_class[cls] = 0.0
            precision_per_class[cls] = 0.0
            continue

        cls_dets.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(cls_dets), dtype=np.float64)
        fp = np.zeros(len(cls_dets), dtype=np.float64)

        gt_for_cls = gt_records[cls]
        for i, (image_id, _score, pred_box) in enumerate(cls_dets):
            gt_entry = gt_for_cls.get(image_id)
            if gt_entry is None:
                fp[i] = 1.0
                continue

            gt_boxes = gt_entry["boxes"]
            overlaps = _bbox_iou_single_to_many(pred_box, gt_boxes)
            max_iou = float(overlaps.max()) if overlaps.size > 0 else 0.0
            jmax = int(overlaps.argmax()) if overlaps.size > 0 else -1

            if max_iou >= iou_threshold and jmax >= 0 and not gt_entry["detected"][jmax]:
                tp[i] = 1.0
                gt_entry["detected"][jmax] = True
            else:
                fp[i] = 1.0

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(max(npos, np.finfo(np.float64).eps))
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap_per_class[cls] = _voc_ap(rec, prec, use_07_metric=use_07_metric)
        recall_per_class[cls] = float(rec[-1]) if rec.size > 0 else 0.0
        precision_per_class[cls] = float(prec[-1]) if prec.size > 0 else 0.0

    fg_classes = np.arange(1, class_count)
    valid_ap_mask = ~np.isnan(ap_per_class[fg_classes])
    valid_ap = ap_per_class[fg_classes][valid_ap_mask]
    map_value = float(valid_ap.mean()) if valid_ap.size > 0 else float("nan")

    matched_gt = 0
    total_gt = 0
    for cls in range(1, class_count):
        for image_id, record in gt_records[cls].items():
            total_gt += int(record["boxes"].shape[0])
            matched_gt += int(record["detected"].sum())
    detector_recall = 100.0 * float(matched_gt) / max(total_gt, 1)

    per_class = []
    for cls in range(1, class_count):
        per_class.append(
            {
                "class_id": cls,
                "class_name": dataset.object_classes[cls],
                "gt": int(gt_count[cls]),
                "detections": int(len(detections[cls])),
                "ap": None if np.isnan(ap_per_class[cls]) else float(ap_per_class[cls]),
                "recall": None if np.isnan(recall_per_class[cls]) else float(recall_per_class[cls] * 100.0),
                "precision": None if np.isnan(precision_per_class[cls]) else float(precision_per_class[cls] * 100.0),
            }
        )

    return {
        "model_path": model_path,
        "vit_model": vit_model,
        "map": map_value,
        "detector_recall": detector_recall,
        "matched_gt": int(matched_gt),
        "gt_boxes": int(total_gt),
        "detections": int(sum(len(v) for v in detections.values())),
        "per_class": per_class,
    }


def compare_runs(baseline, candidate):
    baseline_by_name = {row["class_name"]: row for row in baseline["per_class"]}
    candidate_by_name = {row["class_name"]: row for row in candidate["per_class"]}
    rows = []
    for class_name, base_row in baseline_by_name.items():
        cand_row = candidate_by_name[class_name]
        base_ap = base_row["ap"] if base_row["ap"] is not None else float("nan")
        cand_ap = cand_row["ap"] if cand_row["ap"] is not None else float("nan")
        if np.isnan(base_ap) or np.isnan(cand_ap):
            continue
        rows.append(
            {
                "class_name": class_name,
                "gt": int(base_row["gt"]),
                "baseline_ap": float(base_ap),
                "candidate_ap": float(cand_ap),
                "delta_ap": float(cand_ap - base_ap),
                "baseline_recall": float(base_row["recall"]),
                "candidate_recall": float(cand_row["recall"]),
            }
        )
    rows.sort(key=lambda row: row["delta_ap"], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare per-class detector AP between two checkpoints.")
    parser.add_argument("-data_path", required=True, type=str)
    parser.add_argument("--baseline_model", required=True, type=str)
    parser.add_argument("--candidate_model", required=True, type=str)
    parser.add_argument("--datasize", default="large", type=str)
    parser.add_argument("--backbone", default="vitdet", type=str)
    parser.add_argument("--baseline_vit_model", default="vit_base_patch16_224", type=str)
    parser.add_argument("--candidate_vit_model", default="vit_base_patch16_224.mae", type=str)
    parser.add_argument("--det_threshold", default=0.1, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--max_video_frames", default=-1, type=int)
    parser.add_argument("--iou_threshold", default=0.5, type=float)
    parser.add_argument("--use_07_metric", action="store_true")
    parser.add_argument("--output_json", default=None, type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("baseline_model:", args.baseline_model)
    print("candidate_model:", args.candidate_model)

    baseline = evaluate_checkpoint(
        data_path=args.data_path,
        datasize=args.datasize,
        model_path=args.baseline_model,
        backbone=args.backbone,
        vit_model=args.baseline_vit_model,
        det_threshold=args.det_threshold,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=device,
        max_steps=args.max_steps,
        max_video_frames=args.max_video_frames,
        iou_threshold=args.iou_threshold,
        use_07_metric=args.use_07_metric,
    )
    candidate = evaluate_checkpoint(
        data_path=args.data_path,
        datasize=args.datasize,
        model_path=args.candidate_model,
        backbone=args.backbone,
        vit_model=args.candidate_vit_model,
        det_threshold=args.det_threshold,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=device,
        max_steps=args.max_steps,
        max_video_frames=args.max_video_frames,
        iou_threshold=args.iou_threshold,
        use_07_metric=args.use_07_metric,
    )
    delta_rows = compare_runs(baseline, candidate)

    result = {
        "baseline": baseline,
        "candidate": candidate,
        "delta_rows": delta_rows,
    }

    print("baseline map/recal:", baseline["map"], baseline["detector_recall"])
    print("candidate map/recal:", candidate["map"], candidate["detector_recall"])
    print("top gains:")
    for row in delta_rows[:10]:
        print(
            "  {class_name:24s} delta_ap={delta_ap:+.4f} cand={candidate_ap:.4f} base={baseline_ap:.4f} gt={gt}".format(
                **row
            )
        )
    print("top regressions:")
    for row in list(reversed(delta_rows[-10:])):
        print(
            "  {class_name:24s} delta_ap={delta_ap:+.4f} cand={candidate_ap:.4f} base={baseline_ap:.4f} gt={gt}".format(
                **row
            )
        )

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("wrote", args.output_json)


if __name__ == "__main__":
    main()
