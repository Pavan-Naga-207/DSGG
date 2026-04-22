import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector


def _build_loader(dataset, num_workers, pin_memory):
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


def _load_checkpoint_state_dicts(path, device):
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


def _load_state_dict_flexible(module, incoming_state_dict):
    if incoming_state_dict is None:
        return [], [], []
    current_state = module.state_dict()
    compatible_state = {}
    skipped_shape_keys = []
    for key, value in incoming_state_dict.items():
        if key not in current_state:
            continue
        if current_state[key].shape != value.shape:
            skipped_shape_keys.append((key, tuple(value.shape), tuple(current_state[key].shape)))
            continue
        compatible_state[key] = value
    missing_keys, unexpected_keys = module.load_state_dict(compatible_state, strict=False)
    return missing_keys, unexpected_keys, skipped_shape_keys


def _voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0.0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0
        return float(ap)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def _bbox_iou_single_to_many(box, boxes):
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    ixmin = np.maximum(box[0], boxes[:, 0])
    iymin = np.maximum(box[1], boxes[:, 1])
    ixmax = np.minimum(box[2], boxes[:, 2])
    iymax = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inter = iw * ih
    union = (
        (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
        + (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
        - inter
    )
    return inter / np.maximum(union, np.finfo(np.float32).eps)


def _extract_gt_for_frame(frame_annotation):
    class_to_boxes = {}
    if len(frame_annotation) > 0 and "person_bbox" in frame_annotation[0]:
        person_bbox = np.asarray(frame_annotation[0]["person_bbox"], dtype=np.float32)
        if person_bbox.ndim == 1:
            person_bbox = person_bbox.reshape(1, -1)
        for row in person_bbox:
            if row.shape[0] >= 4:
                class_to_boxes.setdefault(1, []).append(row[:4].astype(np.float32))

    for rel in frame_annotation[1:]:
        if "class" not in rel or "bbox" not in rel:
            continue
        cls = int(rel["class"])
        if cls <= 0:
            continue
        bbox = np.asarray(rel["bbox"], dtype=np.float32).reshape(-1)
        if bbox.shape[0] < 4:
            continue
        class_to_boxes.setdefault(cls, []).append(bbox[:4].astype(np.float32))
    return class_to_boxes


def _find_class_id(name, classes):
    lname = name.strip().lower()
    if not lname:
        return None
    for idx, cname in enumerate(classes):
        if cname.lower() == lname:
            return idx
    for idx, cname in enumerate(classes):
        if lname in cname.lower():
            return idx
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate detector-only mAP@IoU for AG.")
    parser.add_argument("-data_path", required=True, type=str)
    parser.add_argument("-model_path", required=True, type=str)
    parser.add_argument("-datasize", default="large", type=str)
    parser.add_argument("--backbone", default="vitdet", type=str)
    parser.add_argument("--det_threshold", default=0.1, type=float)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--max_video_frames", default=-1, type=int)
    parser.add_argument("--iou_threshold", default=0.5, type=float)
    parser.add_argument("--use_07_metric", action="store_true")
    parser.add_argument(
        "--focus_classes",
        default="person,table,chair,cup/glass/bottle,phone/camera,medicine,food,paper/notebook",
        type=str,
    )
    parser.add_argument("--summary_path", default=None, type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(args.model_path))

    eval_workers = (
        int(os.environ.get("EVAL_NUM_WORKERS", os.environ.get("NUM_WORKERS", "4")))
        if args.num_workers is None
        else int(args.num_workers)
    )
    pin_memory = args.pin_memory or (os.environ.get("PIN_MEMORY", "1") == "1")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("model_path:", args.model_path)
    print("data_path:", args.data_path)
    print("datasize:", args.datasize)
    print("backbone:", args.backbone)
    print("det_threshold:", args.det_threshold)
    print("eval_workers:", eval_workers)
    print("pin_memory:", pin_memory)
    print("max_steps:", args.max_steps)
    print("max_video_frames:", args.max_video_frames)
    print("iou_threshold:", args.iou_threshold)
    print("use_07_metric:", args.use_07_metric)

    ag_test = AG(
        mode="test",
        datasize=args.datasize,
        data_path=args.data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=True,
        backbone=args.backbone,
    )
    dataloader = _build_loader(ag_test, eval_workers, pin_memory)

    object_detector = detector(
        train=False,
        object_classes=ag_test.object_classes,
        use_SUPPLY=True,
        mode="sgdet",
        backbone=args.backbone,
        det_threshold=args.det_threshold,
    ).to(device=device)
    object_detector.eval()

    _, detector_state_dict = _load_checkpoint_state_dicts(args.model_path, device)
    if detector_state_dict is not None:
        d_missing, d_unexpected, d_skipped = _load_state_dict_flexible(object_detector, detector_state_dict)
        print("detector missing/unexpected/skipped:", len(d_missing), len(d_unexpected), len(d_skipped))
        if d_skipped:
            print("detector first skipped:", d_skipped[0][0], d_skipped[0][1], "->", d_skipped[0][2])
    else:
        print("warning: no object_detector_state_dict in checkpoint; using detector defaults")

    class_count = len(ag_test.object_classes)
    selected_videos = len(ag_test) if args.max_steps < 0 else min(len(ag_test), int(args.max_steps))
    print("selected_videos:", selected_videos, "/", len(ag_test))

    gt_records = {cls: {} for cls in range(1, class_count)}
    gt_count = np.zeros(class_count, dtype=np.int64)
    for vid_idx in range(selected_videos):
        frame_names = ag_test.video_list[vid_idx]
        gt_video = ag_test.gt_annotations[vid_idx]
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

    detections = {cls: [] for cls in range(1, class_count)}
    with torch.no_grad():
        for b, data in enumerate(dataloader):
            if args.max_steps > 0 and b >= args.max_steps:
                break

            video_idx = int(data[4])
            frame_names = ag_test.video_list[video_idx]
            gt_annotation = ag_test.gt_annotations[video_idx]

            im_data = data[0].to(device=device, non_blocking=True)
            im_info = data[1].to(device=device, non_blocking=True)
            gt_boxes = data[2].to(device=device, non_blocking=True)
            num_boxes = data[3].to(device=device, non_blocking=True)

            if args.max_video_frames > 0 and im_data.shape[0] > args.max_video_frames:
                im_data = im_data[: args.max_video_frames]
                im_info = im_info[: args.max_video_frames]
                gt_boxes = gt_boxes[: args.max_video_frames]
                num_boxes = num_boxes[: args.max_video_frames]
                frame_names = frame_names[: args.max_video_frames]
                gt_annotation = gt_annotation[: args.max_video_frames]

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

            if (b + 1) % 10 == 0:
                det_count = sum(len(v) for v in detections.values())
                print("processed videos:", b + 1, "detections:", det_count)

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
            if overlaps.size == 0:
                fp[i] = 1.0
                continue

            j = int(np.argmax(overlaps))
            ovmax = float(overlaps[j])
            if ovmax >= args.iou_threshold and not bool(gt_entry["detected"][j]):
                tp[i] = 1.0
                gt_entry["detected"][j] = True
            else:
                fp[i] = 1.0

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / max(float(npos), np.finfo(np.float64).eps)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _voc_ap(rec, prec, use_07_metric=args.use_07_metric)

        ap_per_class[cls] = ap
        recall_per_class[cls] = float(rec[-1]) if rec.size > 0 else 0.0
        precision_per_class[cls] = float(prec[-1]) if prec.size > 0 else 0.0

    fg_classes = np.arange(1, class_count)
    valid_ap_mask = ~np.isnan(ap_per_class[fg_classes])
    valid_ap = ap_per_class[fg_classes][valid_ap_mask]
    map_50 = float(np.mean(valid_ap)) if valid_ap.size > 0 else 0.0

    total_gt = int(np.sum(gt_count[1:]))
    total_det = int(sum(len(v) for v in detections.values()))
    matched_gt = 0
    for cls in range(1, class_count):
        for image_id in gt_records[cls].keys():
            matched_gt += int(np.sum(gt_records[cls][image_id]["detected"]))
    detector_recall = 100.0 * float(matched_gt) / max(total_gt, 1)

    print("==================================================")
    print("DETECTOR mAP SUMMARY (SGDET detector path)")
    print("iou_threshold:", args.iou_threshold)
    print("videos_evaluated:", selected_videos)
    print("gt_boxes:", total_gt)
    print("detections:", total_det)
    print("matched_gt:", matched_gt)
    print("detector_recall@{:.2f}: {:.4f}%".format(args.iou_threshold, detector_recall))
    print("classes_with_gt:", int(valid_ap.size), "/", class_count - 1)
    print("mAP@{:.2f}: {:.4f}".format(args.iou_threshold, map_50))
    print("--------------------------------------------------")

    focus = [x.strip() for x in args.focus_classes.split(",") if x.strip()]
    print("focus class AP/recall/precision:")
    for cname in focus:
        cid = _find_class_id(cname, ag_test.object_classes)
        if cid is None:
            print("  {:24s} not found".format(cname))
            continue
        ap = ap_per_class[cid]
        rec = recall_per_class[cid]
        prec = precision_per_class[cid]
        npos = int(gt_count[cid])
        ap_str = "nan" if np.isnan(ap) else "{:.4f}".format(float(ap))
        rec_str = "nan" if np.isnan(rec) else "{:.2f}%".format(float(rec) * 100.0)
        prec_str = "nan" if np.isnan(prec) else "{:.2f}%".format(float(prec) * 100.0)
        print(
            "  {:24s} gt={:6d} AP={} recall={} precision={}".format(
                ag_test.object_classes[cid], npos, ap_str, rec_str, prec_str
            )
        )

    print("--------------------------------------------------")
    sortable = []
    for cls in range(1, class_count):
        if np.isnan(ap_per_class[cls]):
            continue
        sortable.append((float(ap_per_class[cls]), cls))
    sortable.sort(reverse=True)
    print("top AP classes:")
    for ap, cls in sortable[:10]:
        print("  {:24s} AP={:.4f} gt={}".format(ag_test.object_classes[cls], ap, int(gt_count[cls])))

    print("bottom AP classes (with GT):")
    for ap, cls in sorted(sortable)[:10]:
        print("  {:24s} AP={:.4f} gt={}".format(ag_test.object_classes[cls], ap, int(gt_count[cls])))
    print("==================================================")

    per_class = []
    for cls in range(1, class_count):
        ap_value = ap_per_class[cls]
        recall_value = recall_per_class[cls]
        precision_value = precision_per_class[cls]
        per_class.append(
            {
                "class_id": int(cls),
                "class_name": ag_test.object_classes[cls],
                "gt": int(gt_count[cls]),
                "ap": None if np.isnan(ap_value) else float(ap_value),
                "recall": None if np.isnan(recall_value) else float(recall_value * 100.0),
                "precision": None if np.isnan(precision_value) else float(precision_value * 100.0),
                "detections": int(len(detections[cls])),
            }
        )

    summary = {
        "iou_threshold": float(args.iou_threshold),
        "videos_evaluated": int(selected_videos),
        "gt_boxes": int(total_gt),
        "detections": int(total_det),
        "matched_gt": int(matched_gt),
        "detector_recall": float(detector_recall),
        "classes_with_gt": int(valid_ap.size),
        "num_foreground_classes": int(class_count - 1),
        "map": float(map_50),
        "per_class": per_class,
    }
    print("DETECTOR_SUMMARY_JSON", json.dumps(summary, sort_keys=True))
    if args.summary_path:
        with open(args.summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
