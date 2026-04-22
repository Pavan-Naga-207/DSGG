import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataloader.action_genome_detector import AGDetectorStage1, detector_collate_fn
from lib.object_detector import detector


def _build_loader(dataset, batch_size, num_workers, pin_memory, sampler=None):
    kwargs = {
        'batch_size': batch_size,
        'shuffle': dataset.mode == 'train' and sampler is None,
        'num_workers': num_workers,
        'collate_fn': detector_collate_fn,
        'pin_memory': pin_memory,
        'sampler': sampler,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = int(os.environ.get('PREFETCH_FACTOR', '2'))
    return DataLoader(dataset, **kwargs)


def _split_param_groups(module):
    vit_params = []
    head_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if 'vitdet' in lname and 'backbone' in lname:
            vit_params.append(param)
        else:
            head_params.append(param)
    return vit_params, head_params


def _save_checkpoint(save_path, epoch, object_detector_model, optimizer, best_map):
    payload = {
        'epoch': int(epoch),
        'best_map': float(best_map),
        'object_detector_state_dict': object_detector_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(payload, save_path)


def _run_detector_map_eval(args, checkpoint_path, summary_path):
    cmd = [
        sys.executable,
        '-u',
        'evaluate_detector_map.py',
        '-data_path',
        args.data_path,
        '-model_path',
        checkpoint_path,
        '-datasize',
        args.datasize,
        '--backbone',
        args.backbone,
        '--det_threshold',
        str(args.det_threshold),
        '--num_workers',
        str(args.eval_workers),
        '--max_steps',
        str(args.map_max_steps),
        '--max_video_frames',
        str(args.map_max_video_frames),
        '--iou_threshold',
        str(args.iou_threshold),
        '--focus_classes',
        args.focus_classes,
        '--summary_path',
        summary_path,
    ]
    print('running detector mAP eval:')
    print('  {}'.format(' '.join(cmd)))
    return subprocess.run(cmd, check=False)


def _load_resume(path, object_detector_model, optimizer, device):
    if not path:
        return 0, -1.0
    checkpoint = torch.load(path, map_location=device)
    detector_state_dict = checkpoint.get('object_detector_state_dict')
    if detector_state_dict is None:
        raise RuntimeError('Resume checkpoint missing object_detector_state_dict: {}'.format(path))
    object_detector_model.load_state_dict(detector_state_dict, strict=False)
    if checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    best_map = float(checkpoint.get('best_map', -1.0))
    print('resumed detector stage1 from {} at epoch {}'.format(path, start_epoch))
    return start_epoch, best_map


def _build_repeat_factor_sampler(dataset, threshold, max_repeat):
    sample_weights, class_repeat = dataset.build_repeat_factor_weights(
        threshold=threshold,
        max_repeat=max_repeat,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
    return sampler, class_repeat


def _print_top_class_weights(dataset, class_weights, topk=10):
    if class_weights is None:
        return
    weights = class_weights.detach().cpu().numpy()
    sortable = []
    for class_idx in range(1, len(dataset.object_classes)):
        sortable.append(
            (
                float(weights[class_idx]),
                int(dataset.class_box_hist[class_idx]),
                dataset.object_classes[class_idx],
            )
        )
    sortable.sort(reverse=True)
    print('top weighted detector classes:')
    for weight, count, class_name in sortable[:topk]:
        print('  {:24s} weight={:.4f} gt_boxes={}'.format(class_name, weight, count))


def _print_top_repeat_classes(dataset, class_repeat, topk=10):
    if class_repeat is None:
        return
    repeat_values = class_repeat.detach().cpu().numpy()
    sortable = []
    for class_idx in range(1, len(dataset.object_classes)):
        sortable.append(
            (
                float(repeat_values[class_idx]),
                int(dataset.class_frame_hist[class_idx]),
                dataset.object_classes[class_idx],
            )
        )
    sortable.sort(reverse=True)
    print('top repeat-factor classes:')
    for repeat_factor, frame_count, class_name in sortable[:topk]:
        print('  {:24s} repeat={:.4f} frames={}'.format(class_name, repeat_factor, frame_count))


def _print_focus_class_metrics(summary, focus_classes):
    per_class = summary.get('per_class')
    if not isinstance(per_class, list):
        return
    by_name = {row.get('class_name'): row for row in per_class if isinstance(row, dict)}
    print('focus classes from summary:')
    for class_name in [name.strip() for name in focus_classes.split(',') if name.strip()]:
        row = by_name.get(class_name)
        if row is None:
            print('  {:24s} missing'.format(class_name))
            continue
        ap = row.get('ap')
        recall = row.get('recall')
        precision = row.get('precision')
        gt_count = int(row.get('gt', 0))
        ap_str = 'nan' if ap is None else '{:.4f}'.format(float(ap))
        recall_str = 'nan' if recall is None else '{:.2f}%'.format(float(recall))
        precision_str = 'nan' if precision is None else '{:.2f}%'.format(float(precision))
        print(
            '  {:24s} gt={:6d} AP={} recall={} precision={}'.format(
                class_name,
                gt_count,
                ap_str,
                recall_str,
                precision_str,
            )
        )


def main():
    parser = argparse.ArgumentParser(description='Detector-first ViT stage1 pretraining on Action Genome.')
    parser.add_argument('-data_path', required=True, type=str)
    parser.add_argument('-save_path', required=True, type=str)
    parser.add_argument('-datasize', default='large', type=str)
    parser.add_argument('-nepoch', default=12, type=int)
    parser.add_argument('--backbone', default='vitdet', type=str)
    parser.add_argument('--det_threshold', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=None, type=int)
    parser.add_argument('--eval_workers', default=None, type=int)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--vit_lr', default=float(os.environ.get('VIT_LR', '1e-5')), type=float)
    parser.add_argument('--head_lr', default=float(os.environ.get('HEAD_LR', '1e-4')), type=float)
    parser.add_argument('--weight_decay', default=float(os.environ.get('WEIGHT_DECAY', '1e-4')), type=float)
    parser.add_argument('--grad_clip_norm', default=float(os.environ.get('GRAD_CLIP_NORM', '5.0')), type=float)
    parser.add_argument('--log_every', default=int(os.environ.get('LOG_EVERY', '20')), type=int)
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--map_delta', default=1e-4, type=float)
    parser.add_argument('--max_train_steps', default=-1, type=int)
    parser.add_argument('--map_max_steps', default=80, type=int)
    parser.add_argument('--map_max_video_frames', default=12, type=int)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--cls_loss', default=os.environ.get('DETECTOR_CLS_LOSS', 'ce'), type=str)
    parser.add_argument(
        '--class_weight_mode',
        default=os.environ.get('DETECTOR_CLASS_WEIGHT_MODE', 'none'),
        choices=('none', 'inv_sqrt'),
        type=str,
    )
    parser.add_argument(
        '--class_weight_bg',
        default=float(os.environ.get('DETECTOR_CLASS_WEIGHT_BG', '1.0')),
        type=float,
    )
    parser.add_argument(
        '--sampler',
        default=os.environ.get('DETECTOR_SAMPLER', 'default'),
        choices=('default', 'repeat_factor'),
        type=str,
    )
    parser.add_argument(
        '--repeat_factor_threshold',
        default=float(os.environ.get('DETECTOR_REPEAT_THRESHOLD', '0.01')),
        type=float,
    )
    parser.add_argument(
        '--repeat_factor_max',
        default=float(os.environ.get('DETECTOR_REPEAT_MAX', '4.0')),
        type=float,
    )
    parser.add_argument(
        '--bbox_loss',
        default=os.environ.get('DETECTOR_BBOX_LOSS', 'smoothl1'),
        choices=('smoothl1', 'giou'),
        type=str,
    )
    parser.add_argument(
        '--focus_classes',
        default=os.environ.get(
            'DETECTOR_FOCUS_CLASSES',
            'person,table,chair,cup/glass/bottle,phone/camera,medicine,food,paper/notebook',
        ),
        type=str,
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.environ['DETECTOR_CLS_LOSS'] = args.cls_loss
    os.environ['DETECTOR_BBOX_LOSS'] = args.bbox_loss
    pin_memory = args.pin_memory or (os.environ.get('PIN_MEMORY', '1') == '1')
    train_workers = int(os.environ.get('NUM_WORKERS', '4')) if args.num_workers is None else int(args.num_workers)
    eval_workers = (
        int(os.environ.get('EVAL_NUM_WORKERS', str(max(1, train_workers // 2))))
        if args.eval_workers is None
        else int(args.eval_workers)
    )
    args.eval_workers = eval_workers

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amp_enabled = os.environ.get('AMP', '1') == '1' and device.type == 'cuda'
    amp_dtype_name = os.environ.get('AMP_DTYPE', 'bf16').lower()
    if device.type == 'cuda' and amp_dtype_name in ('bf16', 'bfloat16') and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    print('device:', device)
    print('data_path:', args.data_path)
    print('save_path:', args.save_path)
    print('backbone:', args.backbone)
    print('batch_size:', args.batch_size)
    print('num_workers(train/eval): {}/{}'.format(train_workers, eval_workers))
    print('pin_memory:', pin_memory)
    print('vit_lr/head_lr/wd:', args.vit_lr, args.head_lr, args.weight_decay)
    print('grad_clip_norm:', args.grad_clip_norm)
    print('patience/map_delta:', args.patience, args.map_delta)
    print('max_train_steps:', args.max_train_steps)
    print('map_max_steps/max_video_frames:', args.map_max_steps, args.map_max_video_frames)
    print('VITDET_MODEL:', os.environ.get('VITDET_MODEL', 'vit_base_patch16_224'))
    print('VIT_INPUT_SIZE:', os.environ.get('VIT_INPUT_SIZE', '1024'))
    print('VIT_LSJ_MIN_SCALE:', os.environ.get('VIT_LSJ_MIN_SCALE', '1.0'))
    print('VIT_LSJ_MAX_SCALE:', os.environ.get('VIT_LSJ_MAX_SCALE', '1.0'))
    print('DETECTOR_REINIT_ROI_HEADS:', os.environ.get('DETECTOR_REINIT_ROI_HEADS', '0'))
    print('DETECTOR_BN_MODE:', os.environ.get('DETECTOR_BN_MODE', 'batchnorm'))
    print('detector cls loss:', args.cls_loss)
    print('detector bbox loss:', args.bbox_loss)
    print('detector class weight mode:', args.class_weight_mode)
    print('detector sampler:', args.sampler)
    print('focus classes:', args.focus_classes)

    if args.cls_loss == 'weighted_ce' and args.class_weight_mode == 'none':
        raise ValueError('weighted_ce requires class_weight_mode != none')

    train_dataset = AGDetectorStage1(
        mode='train',
        datasize=args.datasize,
        data_path=args.data_path,
        backbone=args.backbone,
        filter_small_box=True,
    )
    train_sampler = None
    repeat_class_stats = None
    if args.sampler == 'repeat_factor':
        train_sampler, repeat_class_stats = _build_repeat_factor_sampler(
            train_dataset,
            threshold=args.repeat_factor_threshold,
            max_repeat=args.repeat_factor_max,
        )
        _print_top_repeat_classes(train_dataset, repeat_class_stats)
    dataloader_train = _build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=train_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
    )

    object_detector_model = detector(
        train=True,
        object_classes=train_dataset.object_classes,
        use_SUPPLY=True,
        mode='sgdet',
        backbone=args.backbone,
        det_threshold=args.det_threshold,
    ).to(device=device)
    object_detector_model.train()
    object_detector_model.is_train = True
    if args.class_weight_mode == 'inv_sqrt':
        class_weights = train_dataset.build_inverse_sqrt_class_weights(
            background_weight=args.class_weight_bg
        ).to(device=device)
        object_detector_model.set_rcnn_class_weights(class_weights)
        _print_top_class_weights(train_dataset, class_weights)

    vit_params, head_params = _split_param_groups(object_detector_model)
    param_groups = []
    if vit_params:
        param_groups.append({'params': vit_params, 'lr': args.vit_lr})
    if head_params:
        param_groups.append({'params': head_params, 'lr': args.head_lr})
    if not param_groups:
        raise RuntimeError('No trainable parameters found for detector stage1.')
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    start_epoch, best_map = _load_resume(
        path=args.resume,
        object_detector_model=object_detector_model,
        optimizer=optimizer,
        device=device,
    )
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.nepoch):
        object_detector_model.train()
        object_detector_model.is_train = True
        epoch_start = time.time()
        running = {
            'loss': 0.0,
            'rpn_loss_cls': 0.0,
            'rpn_loss_bbox': 0.0,
            'rcnn_loss_cls': 0.0,
            'rcnn_loss_bbox': 0.0,
            'roi_top1': 0.0,
            'roi_top5': 0.0,
            'roi_count': 0,
            'skipped_steps': 0,
            'steps': 0,
        }

        for step, batch in enumerate(dataloader_train):
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break

            images, im_info, gt_boxes, num_boxes, _frame_names = batch
            images = images.to(device=device, non_blocking=True)
            im_info = im_info.to(device=device, non_blocking=True)
            gt_boxes = gt_boxes.to(device=device, non_blocking=True)
            num_boxes = num_boxes.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                stats = object_detector_model.forward_detector_pretrain(images, im_info, gt_boxes, num_boxes)
                if stats is None:
                    loss = None
                else:
                    loss = stats['loss']
            if stats is None:
                running['skipped_steps'] += 1
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(object_detector_model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(object_detector_model.parameters(), args.grad_clip_norm)
                optimizer.step()

            running['steps'] += 1
            running['loss'] += float(loss.detach().item())
            running['rpn_loss_cls'] += float(stats['rpn_loss_cls'].detach().item())
            running['rpn_loss_bbox'] += float(stats['rpn_loss_bbox'].detach().item())
            running['rcnn_loss_cls'] += float(stats['rcnn_loss_cls'].detach().item())
            running['rcnn_loss_bbox'] += float(stats['rcnn_loss_bbox'].detach().item())
            running['roi_top1'] += float(stats['roi_top1'])
            running['roi_top5'] += float(stats['roi_top5'])
            running['roi_count'] += int(stats['roi_count'])
            running['skipped_steps'] += int(stats.get('skipped_chunks', 0))

            if (step + 1) % args.log_every == 0:
                denom = float(max(1, running['steps']))
                elapsed = time.time() - epoch_start
                print(
                    '[stage1] epoch {} step {} loss={:.4f} rpn=({:.4f},{:.4f}) '
                    'rcnn=({:.4f},{:.4f}) roi_top1={:.2f}% roi_top5={:.2f}% '
                    'skipped={} elapsed={:.1f}s'.format(
                        epoch,
                        step + 1,
                        running['loss'] / denom,
                        running['rpn_loss_cls'] / denom,
                        running['rpn_loss_bbox'] / denom,
                        running['rcnn_loss_cls'] / denom,
                        running['rcnn_loss_bbox'] / denom,
                        100.0 * (running['roi_top1'] / denom),
                        100.0 * (running['roi_top5'] / denom),
                        running['skipped_steps'],
                        elapsed,
                    )
                )

        epoch_denom = float(max(1, running['steps']))
        print('=' * 72)
        print(
            'epoch {} train summary: loss={:.4f} rpn=({:.4f},{:.4f}) rcnn=({:.4f},{:.4f}) '
            'roi_top1={:.2f}% roi_top5={:.2f}% steps={} skipped={} roi_count={}'.format(
                epoch,
                running['loss'] / epoch_denom,
                running['rpn_loss_cls'] / epoch_denom,
                running['rpn_loss_bbox'] / epoch_denom,
                running['rcnn_loss_cls'] / epoch_denom,
                running['rcnn_loss_bbox'] / epoch_denom,
                100.0 * (running['roi_top1'] / epoch_denom),
                100.0 * (running['roi_top5'] / epoch_denom),
                running['steps'],
                running['skipped_steps'],
                running['roi_count'],
            )
        )

        last_ckpt_path = os.path.join(args.save_path, 'detector_stage1_last.pth')
        epoch_ckpt_path = os.path.join(args.save_path, 'detector_stage1_epoch_{:02d}.pth'.format(epoch))
        _save_checkpoint(last_ckpt_path, epoch, object_detector_model, optimizer, best_map)
        shutil.copyfile(last_ckpt_path, epoch_ckpt_path)

        improved = False
        if (epoch + 1) % args.eval_every == 0:
            summary_path = os.path.join(args.save_path, 'detector_map_epoch_{:02d}.json'.format(epoch))
            result = _run_detector_map_eval(args, epoch_ckpt_path, summary_path)
            if result.returncode != 0:
                print('detector mAP eval failed with exit code {}'.format(result.returncode))
            elif os.path.isfile(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                current_map = float(summary.get('map', 0.0))
                current_recall = float(summary.get('detector_recall', 0.0))
                print(
                    'epoch {} detector eval: mAP@{:.2f}={:.4f} recall={:.4f}%'.format(
                        epoch,
                        args.iou_threshold,
                        current_map,
                        current_recall,
                    )
                )
                _print_focus_class_metrics(summary, args.focus_classes)
                if current_map > best_map + args.map_delta:
                    best_map = current_map
                    improved = True
                    best_ckpt_path = os.path.join(args.save_path, 'detector_stage1_best.pth')
                    shutil.copyfile(epoch_ckpt_path, best_ckpt_path)
                    print('new best detector checkpoint -> {}'.format(best_ckpt_path))

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print('best_map_so_far:', best_map)
        print('epochs_without_improvement:', epochs_without_improvement)
        print('=' * 72)

        if epochs_without_improvement >= args.patience:
            print(
                'early stopping detector stage1 after {} epochs without mAP improvement'.format(
                    epochs_without_improvement
                )
            )
            break


if __name__ == '__main__':
    main()
