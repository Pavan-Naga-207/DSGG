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


def _load_checkpoint_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError('Unsupported checkpoint format at {}'.format(path))

    # Handle checkpoints saved from DDP wrappers.
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict


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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Evaluation device:', device)
    print('mode:', conf.mode)
    print('datasize:', conf.datasize)
    print('data_path:', conf.data_path)
    print('model_path:', conf.model_path)
    print('eval_workers:', eval_workers)
    print('pin_memory:', pin_memory)
    print('max_test_steps:', max_test_steps)
    print('constraints:', constraints)

    ag_test = AG(
        mode='test',
        datasize=conf.datasize,
        data_path=conf.data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False if conf.mode == 'predcls' else True,
    )
    dataloader = _build_loader(ag_test, eval_workers, pin_memory)

    object_detector = detector(
        train=False,
        object_classes=ag_test.object_classes,
        use_SUPPLY=True,
        mode=conf.mode,
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

    state_dict = _load_checkpoint_state_dict(conf.model_path, device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print('warning: missing keys count:', len(missing_keys))
    if unexpected_keys:
        print('warning: unexpected keys count:', len(unexpected_keys))
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
