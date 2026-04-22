import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.action_genome import AG, cuda_collate_fn
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.sttran import STTran


def _build_loader(dataset, num_workers, pin_memory):
    kwargs = {
        'shuffle': False,
        'num_workers': num_workers,
        'collate_fn': cuda_collate_fn,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = int(os.environ.get('PREFETCH_FACTOR', '2'))
    return DataLoader(dataset, **kwargs)


def _parse_constraints(raw_constraints):
    parsed = []
    for token in raw_constraints.split(','):
        key = token.strip().lower()
        if not key:
            continue
        if key in ('with', 'semi', 'no'):
            parsed.append(key)
    if not parsed:
        parsed = ['with']
    return parsed


def _strip_ddp_prefix(state_dict):
    if state_dict is None:
        return None
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict


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


def _load_checkpoint_state_dicts(path, device):
    checkpoint = torch.load(path, map_location=device)
    model_state_dict = None
    detector_state_dict = None

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            model_state_dict = checkpoint
        detector_state_dict = checkpoint.get('object_detector_state_dict')
    else:
        raise RuntimeError('Unsupported checkpoint format at {}'.format(path))

    return _strip_ddp_prefix(model_state_dict), _strip_ddp_prefix(detector_state_dict)


def main():
    np.set_printoptions(precision=4)
    conf = Config()

    if not conf.model_path:
        raise ValueError('model_path is required. Pass -model_path /path/to/checkpoint.pth')
    if not os.path.isfile(conf.model_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(conf.model_path))

    eval_workers = int(os.environ.get('EVAL_NUM_WORKERS', os.environ.get('NUM_WORKERS', '4')))
    pin_memory = os.environ.get('PIN_MEMORY', '1') == '1'
    max_test_steps = int(os.environ.get('MAX_TEST_STEPS', '-1'))
    constraints = _parse_constraints(os.environ.get('EVAL_CONSTRAINTS', 'with,semi,no'))
    debug_detector_once = os.environ.get('DEBUG_DETECTOR_ONCE', '0') == '1'
    debug_detector_batch = int(os.environ.get('DEBUG_DETECTOR_BATCH', '0'))
    debug_detector_topk = int(os.environ.get('DEBUG_DETECTOR_TOPK', '20'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Evaluation device:', device)
    print('mode:', conf.mode)
    print('datasize:', conf.datasize)
    print('data_path:', conf.data_path)
    print('model_path:', conf.model_path)
    print('backbone:', conf.backbone)
    print('eval_workers:', eval_workers)
    print('pin_memory:', pin_memory)
    print('max_test_steps:', max_test_steps)
    print('constraints:', constraints)
    print('debug_detector_once:', debug_detector_once)
    if debug_detector_once:
        print('debug_detector_batch:', debug_detector_batch)
        print('debug_detector_topk:', debug_detector_topk)

    ag_test = AG(
        mode='test',
        datasize=conf.datasize,
        data_path=conf.data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False if conf.mode == 'predcls' else True,
        backbone=conf.backbone,
    )
    dataloader = _build_loader(ag_test, eval_workers, pin_memory)

    object_detector = detector(
        train=False,
        object_classes=ag_test.object_classes,
        use_SUPPLY=True,
        mode=conf.mode,
        backbone=conf.backbone,
        det_threshold=conf.det_threshold,
    ).to(device=device)
    object_detector.eval()

    model = STTran(
        mode=conf.mode,
        attention_class_num=len(ag_test.attention_relationships),
        spatial_class_num=len(ag_test.spatial_relationships),
        contact_class_num=len(ag_test.contacting_relationships),
        obj_classes=ag_test.object_classes,
        enc_layer_num=conf.enc_layer,
        dec_layer_num=conf.dec_layer,
    ).to(device=device)
    model.eval()

    model_state_dict, detector_state_dict = _load_checkpoint_state_dicts(conf.model_path, device)
    if detector_state_dict is not None:
        det_missing_keys, det_unexpected_keys, det_skipped_shapes = _load_state_dict_flexible(
            object_detector, detector_state_dict
        )
        if det_missing_keys:
            print('warning: detector missing keys count:', len(det_missing_keys))
        if det_unexpected_keys:
            print('warning: detector unexpected keys count:', len(det_unexpected_keys))
        if det_skipped_shapes:
            print('warning: detector skipped shape keys count:', len(det_skipped_shapes))
            print(
                'warning: detector first skipped key:',
                det_skipped_shapes[0][0],
                det_skipped_shapes[0][1],
                '->',
                det_skipped_shapes[0][2],
            )

    missing_keys, unexpected_keys, model_skipped_shapes = _load_state_dict_flexible(model, model_state_dict)
    if missing_keys:
        print('warning: missing keys count:', len(missing_keys))
    if unexpected_keys:
        print('warning: unexpected keys count:', len(unexpected_keys))
    if model_skipped_shapes:
        print('warning: model skipped shape keys count:', len(model_skipped_shapes))
    print('*' * 50)
    print('Loaded checkpoint: {}'.format(conf.model_path))

    evaluators = []
    for constraint in constraints:
        if constraint == 'semi':
            evaluator = BasicSceneGraphEvaluator(
                mode=conf.mode,
                AG_object_classes=ag_test.object_classes,
                AG_all_predicates=ag_test.relationship_classes,
                AG_attention_predicates=ag_test.attention_relationships,
                AG_spatial_predicates=ag_test.spatial_relationships,
                AG_contacting_predicates=ag_test.contacting_relationships,
                iou_threshold=0.5,
                constraint='semi',
                semithreshold=0.9,
            )
        else:
            evaluator = BasicSceneGraphEvaluator(
                mode=conf.mode,
                AG_object_classes=ag_test.object_classes,
                AG_all_predicates=ag_test.relationship_classes,
                AG_attention_predicates=ag_test.attention_relationships,
                AG_spatial_predicates=ag_test.spatial_relationships,
                AG_contacting_predicates=ag_test.contacting_relationships,
                iou_threshold=0.5,
                constraint=constraint,
            )
        evaluators.append((constraint, evaluator))

    with torch.no_grad():
        for b, data in enumerate(dataloader):
            if max_test_steps > 0 and b >= max_test_steps:
                break

            im_data = data[0].to(device=device, non_blocking=True)
            im_info = data[1].to(device=device, non_blocking=True)
            gt_boxes = data[2].to(device=device, non_blocking=True)
            num_boxes = data[3].to(device=device, non_blocking=True)
            gt_annotation = ag_test.gt_annotations[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

            if debug_detector_once and b == debug_detector_batch:
                print('\n' + '!' * 50)
                print('DEBUG: RAW DETECTOR OUTPUT FOR BATCH {}'.format(b))
                print('!' * 50)

                print('Keys in entry dict:', sorted(list(entry.keys())))

                labels_key = 'pred_labels' if 'pred_labels' in entry else ('labels' if 'labels' in entry else None)
                boxes_key = 'pred_boxes' if 'pred_boxes' in entry else ('boxes' if 'boxes' in entry else None)
                scores_key = 'pred_scores' if 'pred_scores' in entry else ('scores' if 'scores' in entry else None)

                if boxes_key is not None and torch.is_tensor(entry[boxes_key]) and entry[boxes_key].numel() > 0:
                    boxes_tensor = entry[boxes_key].detach()
                    if boxes_tensor.dim() == 2 and boxes_tensor.size(1) >= 5:
                        frame_ids = boxes_tensor[:, 0].long().cpu()
                        max_frame = int(frame_ids.max().item()) if frame_ids.numel() > 0 else -1
                        per_frame = torch.bincount(frame_ids, minlength=max_frame + 1) if max_frame >= 0 else torch.zeros(0, dtype=torch.long)
                        print('\nDetected boxes per frame (post-NMS):')
                        print(per_frame.tolist())
                        print('total_detected_boxes:', int(per_frame.sum().item()))

                gt_labels = []
                for frame_anno in gt_annotation:
                    for item in frame_anno:
                        if 'person_bbox' in item:
                            gt_labels.append(1)
                        elif 'class' in item:
                            gt_labels.append(int(item['class']))
                if gt_labels:
                    gt_preview = gt_labels[:debug_detector_topk]
                    gt_names = [
                        ag_test.object_classes[idx] if 0 <= idx < len(ag_test.object_classes) else 'unknown'
                        for idx in gt_preview
                    ]
                    print('\nGT labels (first {}):'.format(len(gt_preview)))
                    print(gt_preview)
                    print('GT label names (first {}):'.format(len(gt_names)))
                    print(gt_names)
                else:
                    print('\nGT labels: none parsed for this batch.')

                if labels_key is not None:
                    labels = entry[labels_key]
                    if torch.is_tensor(labels):
                        labels_cpu = labels.detach().cpu()
                        print('\n{} dtype={} shape={}'.format(labels_key, labels_cpu.dtype, tuple(labels_cpu.shape)))
                        pred_preview = labels_cpu[:debug_detector_topk]
                        print(pred_preview)
                        if labels_key in ('pred_labels', 'labels'):
                            pred_names = [
                                ag_test.object_classes[int(idx)] if 0 <= int(idx) < len(ag_test.object_classes) else 'unknown'
                                for idx in pred_preview.tolist()
                            ]
                            print('pred label names (first {}):'.format(len(pred_names)))
                            print(pred_names)
                    else:
                        print('\n{} (non-tensor): {}'.format(labels_key, type(labels)))
                        print(labels)
                else:
                    print('\nNo label key found (expected pred_labels or labels).')

                if boxes_key is not None:
                    boxes = entry[boxes_key]
                    if torch.is_tensor(boxes):
                        boxes_cpu = boxes.detach().cpu()
                        print('\n{} dtype={} shape={}'.format(boxes_key, boxes_cpu.dtype, tuple(boxes_cpu.shape)))
                        print(boxes_cpu[:5])
                    else:
                        print('\n{} (non-tensor): {}'.format(boxes_key, type(boxes)))
                        print(boxes)
                else:
                    print('\nNo box key found (expected pred_boxes or boxes).')

                if scores_key is not None:
                    scores = entry[scores_key]
                    if torch.is_tensor(scores):
                        scores_cpu = scores.detach().cpu()
                        print('\n{} dtype={} shape={}'.format(scores_key, scores_cpu.dtype, tuple(scores_cpu.shape)))
                        print(scores_cpu[:debug_detector_topk])
                    else:
                        print('\n{} (non-tensor): {}'.format(scores_key, type(scores)))
                        print(scores)
                else:
                    print('\nNo score key found (expected pred_scores or scores).')

                print('\nEXITING DEBUG SCRIPT...')
                raise SystemExit(0)

            pred = model(entry)

            for _, evaluator in evaluators:
                evaluator.evaluate_scene_graph(gt_annotation, dict(pred))

            if b > 0 and b % 500 == 0:
                print('evaluated batches:', b)

    print('-----------', flush=True)
    for constraint, evaluator in evaluators:
        print('======================{}============================'.format(constraint))
        evaluator.print_stats()


if __name__ == '__main__':
    main()
