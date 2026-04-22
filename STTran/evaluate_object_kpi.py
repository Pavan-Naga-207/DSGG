import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.action_genome import AG, cuda_collate_fn
from lib.fpn.box_utils import center_size
from lib.object_detector import detector
from lib.sttran import STTran


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


def _topk_correct(dist_logits, gt_labels, k):
    topk = torch.topk(dist_logits, k=min(k, dist_logits.shape[1]), dim=1).indices + 1
    return (topk == gt_labels.unsqueeze(1)).any(dim=1)


def _safe_pct(numer, denom):
    if denom <= 0:
        return 0.0
    return 100.0 * float(numer) / float(denom)


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


def _top_hist(hist, classes, topn=10):
    pairs = [(idx, int(v)) for idx, v in enumerate(hist) if int(v) > 0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [(classes[idx], cnt) for idx, cnt in pairs[:topn]]


def main():
    parser = argparse.ArgumentParser(description="Evaluate object-only KPIs on GT boxes (SGCLS path).")
    parser.add_argument("-data_path", required=True, type=str)
    parser.add_argument("-model_path", required=True, type=str)
    parser.add_argument("-datasize", default="large", type=str)
    parser.add_argument("--backbone", default="vitdet", type=str)
    parser.add_argument("--det_threshold", default=0.1, type=float)
    parser.add_argument("--enc_layer", default=1, type=int)
    parser.add_argument("--dec_layer", default=3, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--sample_count", default=20, type=int)
    parser.add_argument(
        "--focus_classes",
        default="person,table,chair,cup/glass/bottle,phone/camera",
        type=str,
    )
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
        mode="sgcls",
        backbone=args.backbone,
        det_threshold=args.det_threshold,
    ).to(device=device)
    object_detector.eval()

    model = STTran(
        mode="sgcls",
        attention_class_num=len(ag_test.attention_relationships),
        spatial_class_num=len(ag_test.spatial_relationships),
        contact_class_num=len(ag_test.contacting_relationships),
        obj_classes=ag_test.object_classes,
        enc_layer_num=args.enc_layer,
        dec_layer_num=args.dec_layer,
    ).to(device=device)
    model.eval()

    model_state_dict, detector_state_dict = _load_checkpoint_state_dicts(args.model_path, device)
    if detector_state_dict is not None:
        d_missing, d_unexpected, d_skipped = _load_state_dict_flexible(object_detector, detector_state_dict)
        print("detector missing/unexpected/skipped:", len(d_missing), len(d_unexpected), len(d_skipped))
    m_missing, m_unexpected, m_skipped = _load_state_dict_flexible(model, model_state_dict)
    print("model missing/unexpected/skipped:", len(m_missing), len(m_unexpected), len(m_skipped))
    if d_skipped:
        print("detector first skipped:", d_skipped[0][0], d_skipped[0][1], "->", d_skipped[0][2])

    class_count = len(ag_test.object_classes)
    gt_hist = np.zeros(class_count, dtype=np.int64)
    det_hist = np.zeros(class_count, dtype=np.int64)
    dec_hist = np.zeros(class_count, dtype=np.int64)
    det_correct_per_class = np.zeros(class_count, dtype=np.int64)
    dec_correct_per_class = np.zeros(class_count, dtype=np.int64)
    total_per_class = np.zeros(class_count, dtype=np.int64)

    total_boxes = 0
    det_top1_correct = 0
    det_top5_correct = 0
    dec_top1_correct = 0
    dec_top5_correct = 0
    samples: List[Tuple[str, str, str]] = []

    with torch.no_grad():
        for b, data in enumerate(dataloader):
            if args.max_steps > 0 and b >= args.max_steps:
                break

            im_data = data[0].to(device=device, non_blocking=True)
            im_info = data[1].to(device=device, non_blocking=True)
            gt_boxes = data[2].to(device=device, non_blocking=True)
            num_boxes = data[3].to(device=device, non_blocking=True)
            gt_annotation = ag_test.gt_annotations[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            if entry["labels"].numel() == 0:
                continue

            gt = entry["labels"].long()
            det_dist = entry["distribution"].float()
            det_top1 = det_dist.argmax(dim=1) + 1
            det_top5_hit = _topk_correct(det_dist, gt, k=5)

            obj_embed = det_dist @ model.object_classifier.obj_embed.weight
            pos_embed = model.object_classifier.pos_embed(center_size(entry["boxes"][:, 1:]))
            obj_features = torch.cat((entry["features"], obj_embed, pos_embed), dim=1)
            dec_logits = model.object_classifier.decoder_lin(obj_features).float()
            dec_dist = torch.softmax(dec_logits[:, 1:], dim=1)
            dec_top1 = dec_dist.argmax(dim=1) + 1
            dec_top5_hit = _topk_correct(dec_dist, gt, k=5)

            n = int(gt.numel())
            total_boxes += n
            det_top1_correct += int((det_top1 == gt).sum().item())
            det_top5_correct += int(det_top5_hit.sum().item())
            dec_top1_correct += int((dec_top1 == gt).sum().item())
            dec_top5_correct += int(dec_top5_hit.sum().item())

            gt_np = gt.detach().cpu().numpy()
            det_np = det_top1.detach().cpu().numpy()
            dec_np = dec_top1.detach().cpu().numpy()
            for i in range(n):
                g = int(gt_np[i])
                d = int(det_np[i])
                c = int(dec_np[i])
                if 0 <= g < class_count:
                    gt_hist[g] += 1
                    total_per_class[g] += 1
                if 0 <= d < class_count:
                    det_hist[d] += 1
                if 0 <= c < class_count:
                    dec_hist[c] += 1
                if d == g and 0 <= g < class_count:
                    det_correct_per_class[g] += 1
                if c == g and 0 <= g < class_count:
                    dec_correct_per_class[g] += 1
                if len(samples) < args.sample_count:
                    gt_name = ag_test.object_classes[g] if 0 <= g < class_count else "UNK"
                    det_name = ag_test.object_classes[d] if 0 <= d < class_count else "UNK"
                    dec_name = ag_test.object_classes[c] if 0 <= c < class_count else "UNK"
                    samples.append((gt_name, det_name, dec_name))

            if (b + 1) % 50 == 0:
                print("processed batches:", b + 1, "boxes:", total_boxes)

    print("==================================================")
    print("OBJECT KPI SUMMARY (SGCLS GT-box path)")
    print("boxes:", total_boxes)
    print("detector_top1: {:.4f}%".format(_safe_pct(det_top1_correct, total_boxes)))
    print("detector_top5: {:.4f}%".format(_safe_pct(det_top5_correct, total_boxes)))
    print("decoder_top1: {:.4f}%".format(_safe_pct(dec_top1_correct, total_boxes)))
    print("decoder_top5: {:.4f}%".format(_safe_pct(dec_top5_correct, total_boxes)))

    print("--------------------------------------------------")
    print("top predicted classes (detector):")
    for name, cnt in _top_hist(det_hist, ag_test.object_classes, topn=10):
        print("  {:24s} {:7d} ({:.3f}%)".format(name, cnt, _safe_pct(cnt, total_boxes)))
    print("top predicted classes (decoder):")
    for name, cnt in _top_hist(dec_hist, ag_test.object_classes, topn=10):
        print("  {:24s} {:7d} ({:.3f}%)".format(name, cnt, _safe_pct(cnt, total_boxes)))

    focus = [x.strip() for x in args.focus_classes.split(",") if x.strip()]
    print("--------------------------------------------------")
    print("focus class accuracy:")
    for cname in focus:
        cid = _find_class_id(cname, ag_test.object_classes)
        if cid is None:
            print("  {:24s} not found".format(cname))
            continue
        tot = int(total_per_class[cid])
        d_ok = int(det_correct_per_class[cid])
        c_ok = int(dec_correct_per_class[cid])
        print(
            "  {:24s} gt={:6d} det_top1={:7.3f}% dec_top1={:7.3f}%".format(
                ag_test.object_classes[cid],
                tot,
                _safe_pct(d_ok, tot),
                _safe_pct(c_ok, tot),
            )
        )

    print("--------------------------------------------------")
    print("sample GT vs predictions (first {}):".format(len(samples)))
    for i, (g, d, c) in enumerate(samples):
        print("  {:02d}. GT={:<24s} DET={:<24s} DEC={}".format(i + 1, g, d, c))
    print("==================================================")


if __name__ == "__main__":
    main()
