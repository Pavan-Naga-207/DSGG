import os
import re
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloader.action_genome import AG, cuda_collate_fn
from lib.AdamW import AdamW
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.sttran import STTran

np.set_printoptions(precision=3)


def _setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    pg_timeout_seconds = int(os.environ.get('DDP_TIMEOUT_SEC', '7200'))
    pg_timeout = timedelta(seconds=pg_timeout_seconds)

    if is_distributed:
        master_addr = os.environ.get('MASTER_ADDR')
        master_port = os.environ.get('MASTER_PORT', '29500')
        if master_addr:
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://{}:{}'.format(master_addr, master_port),
                rank=rank,
                world_size=world_size,
                timeout=pg_timeout,
            )
        else:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=rank,
                world_size=world_size,
                timeout=pg_timeout,
            )

    return is_distributed, rank, world_size, local_rank, device, pg_timeout_seconds


def _build_loader(dataset, sampler, shuffle, num_workers, pin_memory):
    kwargs = {
        'collate_fn': cuda_collate_fn,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'shuffle': shuffle,
        'sampler': sampler,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = int(os.environ.get('PREFETCH_FACTOR', '2'))
    return DataLoader(dataset, **kwargs)


def _optimizer_for(params, conf):
    if conf.optimizer == 'adamw':
        use_fused = os.environ.get('FUSED_ADAMW', '1') == '1'
        if use_fused:
            try:
                return optim.AdamW(params, lr=conf.lr, fused=True)
            except Exception:
                pass
        return AdamW(params, lr=conf.lr)
    if conf.optimizer == 'adam':
        return optim.Adam(params, lr=conf.lr)
    if conf.optimizer == 'sgd':
        return optim.SGD(params, lr=conf.lr, momentum=0.9, weight_decay=0.01)
    raise ValueError('Unknown optimizer {}'.format(conf.optimizer))


def _unwrap(module):
    return module.module if hasattr(module, 'module') else module


def _split_detector_params(detector_module):
    vit_params = []
    detector_task_params = []
    for name, param in detector_module.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if 'vitdet' in lname and 'backbone' in lname:
            vit_params.append(param)
        else:
            detector_task_params.append(param)
    return vit_params, detector_task_params


def _build_train_optimizer(model, object_detector, conf, train_detector):
    if not train_detector:
        optimizer = _optimizer_for(model.parameters(), conf)
        return optimizer, {
            'train_detector': False,
            'vit_lr': None,
            'task_lr': conf.lr,
            'vit_param_tensors': 0,
            'task_param_tensors': sum(1 for _ in model.parameters()),
        }

    detector_module = _unwrap(object_detector)
    vit_params, detector_task_params = _split_detector_params(detector_module)
    model_params = [p for p in model.parameters() if p.requires_grad]
    task_params = detector_task_params + model_params

    vit_lr = float(os.environ.get('VIT_LR', str(conf.lr)))
    task_lr = float(os.environ.get('TASK_LR', '1e-4'))
    weight_decay = float(os.environ.get('WEIGHT_DECAY', '1e-4'))

    param_groups = []
    if vit_params:
        param_groups.append({'params': vit_params, 'lr': vit_lr})
    if task_params:
        param_groups.append({'params': task_params, 'lr': task_lr})

    if not param_groups:
        raise RuntimeError('No trainable parameters found for differential optimizer.')

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer, {
        'train_detector': True,
        'vit_lr': vit_lr,
        'task_lr': task_lr,
        'weight_decay': weight_decay,
        'vit_param_tensors': len(vit_params),
        'task_param_tensors': len(task_params),
    }


def _params_for_grad_clip(optimizer):
    params = []
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.requires_grad:
                params.append(param)
    return params


def _strip_ddp_prefix(state_dict):
    if state_dict is None:
        return None
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def _infer_epoch_from_path(path):
    name = os.path.basename(path)
    match = re.search(r'sttran_epoch_(\d+)\.pth$', name)
    if match is None:
        return None
    return int(match.group(1))


def _load_resume_checkpoint(conf, model, object_detector, optimizer, scheduler, device, is_main):
    if not conf.ckpt:
        return 0
    ckpt_path = conf.ckpt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError('Resume checkpoint not found: {}'.format(ckpt_path))

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_state_dict = None
    detector_state_dict = None
    optimizer_state_dict = None
    scheduler_state_dict = None
    ckpt_epoch = None

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            # Legacy checkpoint format: plain model state_dict
            model_state_dict = checkpoint
        detector_state_dict = checkpoint.get('object_detector_state_dict')
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        scheduler_state_dict = checkpoint.get('scheduler_state_dict')
        ckpt_epoch = checkpoint.get('epoch')
    else:
        raise RuntimeError('Unsupported checkpoint format at {}'.format(ckpt_path))

    model_state_dict = _strip_ddp_prefix(model_state_dict)
    detector_state_dict = _strip_ddp_prefix(detector_state_dict)

    model_module = _unwrap(model)
    detector_module = _unwrap(object_detector)

    missing_keys, unexpected_keys = model_module.load_state_dict(model_state_dict, strict=False)
    if is_main:
        print('resume model missing_keys:', len(missing_keys))
        print('resume model unexpected_keys:', len(unexpected_keys))

    if detector_state_dict is not None:
        det_missing_keys, det_unexpected_keys = detector_module.load_state_dict(detector_state_dict, strict=False)
        if is_main:
            print('resume detector missing_keys:', len(det_missing_keys))
            print('resume detector unexpected_keys:', len(det_unexpected_keys))

    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            if is_main:
                print('resume optimizer: loaded')
        except Exception as exc:
            if is_main:
                print('resume optimizer: skipped ({})'.format(exc))
    if scheduler is not None and scheduler_state_dict is not None:
        try:
            scheduler.load_state_dict(scheduler_state_dict)
            if is_main:
                print('resume scheduler: loaded')
        except Exception as exc:
            if is_main:
                print('resume scheduler: skipped ({})'.format(exc))

    if ckpt_epoch is None:
        ckpt_epoch = _infer_epoch_from_path(ckpt_path)
    start_epoch = int(ckpt_epoch) + 1 if ckpt_epoch is not None else 0
    if is_main:
        print('resuming from checkpoint:', ckpt_path)
        print('resume start_epoch:', start_epoch)
    return start_epoch


def main():
    conf = Config()
    is_distributed, rank, world_size, local_rank, gpu_device, pg_timeout_seconds = _setup_distributed()
    is_main = rank == 0

    train_workers = int(os.environ.get('NUM_WORKERS', '4'))
    eval_workers = int(os.environ.get('EVAL_NUM_WORKERS', str(max(1, train_workers // 2))))
    pin_memory = os.environ.get('PIN_MEMORY', '1') == '1'

    amp_enabled = os.environ.get('AMP', '1') == '1' and gpu_device.type == 'cuda'
    amp_dtype_name = os.environ.get('AMP_DTYPE', 'bf16').lower()
    if gpu_device.type == 'cuda' and amp_dtype_name in ('bf16', 'bfloat16') and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    if conf.max_train_steps is not None:
        max_train_steps = int(conf.max_train_steps)
    else:
        max_train_steps = int(os.environ.get('MAX_TRAIN_STEPS', '-1'))
    if conf.max_test_steps is not None:
        max_test_steps = int(conf.max_test_steps)
    else:
        max_test_steps = int(os.environ.get('MAX_TEST_STEPS', '-1'))
    run_eval = os.environ.get('RUN_EVAL', '0') == '1'
    ckpt_every = max(1, int(os.environ.get('CKPT_EVERY', '2')))
    eval_every = max(1, int(os.environ.get('EVAL_EVERY', '1')))
    train_detector_default = '1' if conf.backbone.lower() == 'vitdet' else '0'
    train_detector = os.environ.get('TRAIN_DETECTOR', train_detector_default) == '1'
    if conf.mode != 'sgdet':
        train_detector = False
    max_video_frames_default = '24' if train_detector else '-1'
    max_video_frames = int(os.environ.get('MAX_VIDEO_FRAMES', max_video_frames_default))

    # Rank-0-only eval inside DDP causes other ranks to wait in collectives and can
    # trigger NCCL watchdog timeouts on long validation loops.
    if run_eval and is_distributed:
        if is_main:
            print('RUN_EVAL=1 requested with DDP; disabling in-training eval for stability. Use evaluate_only.py.')
        run_eval = False

    if is_main:
        print('The CKPT saved here:', conf.save_path)
        if not os.path.exists(conf.save_path):
            os.mkdir(conf.save_path)
        print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
        for k in conf.args:
            print(k, ':', conf.args[k])
        print('distributed: {} world_size={} rank={} local_rank={}'.format(is_distributed, world_size, rank, local_rank))
        print('num_workers(train/eval): {}/{}'.format(train_workers, eval_workers))
        print('pin_memory: {}'.format(pin_memory))
        print('amp_enabled: {} amp_dtype: {}'.format(amp_enabled, amp_dtype))
        print('ddp_timeout_sec: {}'.format(pg_timeout_seconds))
        print('max_train_steps: {}  max_test_steps: {}'.format(max_train_steps, max_test_steps))
        print('run_eval: {}  ckpt_every: {}  eval_every: {}'.format(run_eval, ckpt_every, eval_every))
        print('train_detector: {}'.format(train_detector))
        print('max_video_frames: {}'.format(max_video_frames))

    AG_dataset_train = AG(
        mode='train',
        datasize=conf.datasize,
        data_path=conf.data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False if conf.mode == 'predcls' else True,
    )
    AG_dataset_test = None
    if run_eval:
        AG_dataset_test = AG(
            mode='test',
            datasize=conf.datasize,
            data_path=conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == 'predcls' else True,
        )
    elif is_main:
        print('In-training evaluation disabled; skipping test dataset initialization.')

    train_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(
            AG_dataset_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
    dataloader_train = _build_loader(
        AG_dataset_train,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=train_workers,
        pin_memory=pin_memory,
    )
    dataloader_test = None
    if run_eval:
        dataloader_test = _build_loader(
            AG_dataset_test,
            sampler=None,
            shuffle=False,
            num_workers=eval_workers,
            pin_memory=pin_memory,
        )

    object_detector = detector(
        train=True,
        object_classes=AG_dataset_train.object_classes,
        use_SUPPLY=True,
        mode=conf.mode,
        backbone=conf.backbone,
        det_threshold=conf.det_threshold,
    ).to(device=gpu_device)
    if is_distributed and train_detector:
        object_detector = DDP(
            object_detector,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    if train_detector:
        object_detector.train()
    else:
        object_detector.eval()

    model = STTran(
        mode=conf.mode,
        attention_class_num=len(AG_dataset_train.attention_relationships),
        spatial_class_num=len(AG_dataset_train.spatial_relationships),
        contact_class_num=len(AG_dataset_train.contacting_relationships),
        obj_classes=AG_dataset_train.object_classes,
        enc_layer_num=conf.enc_layer,
        dec_layer_num=conf.dec_layer,
    ).to(device=gpu_device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    evaluator = None
    if run_eval:
        evaluator = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset_train.object_classes,
            AG_all_predicates=AG_dataset_train.relationship_classes,
            AG_attention_predicates=AG_dataset_train.attention_relationships,
            AG_spatial_predicates=AG_dataset_train.spatial_relationships,
            AG_contacting_predicates=AG_dataset_train.contacting_relationships,
            iou_threshold=0.5,
            constraint='with',
        )

    # loss function, default Multi-label margin loss
    ce_loss = nn.CrossEntropyLoss()
    if conf.bce_loss:
        bce_loss = nn.BCELoss()
    else:
        mlm_loss = nn.MultiLabelMarginLoss()

    optimizer, optimizer_info = _build_train_optimizer(
        model=model,
        object_detector=object_detector,
        conf=conf,
        train_detector=train_detector,
    )
    clip_params = _params_for_grad_clip(optimizer)
    if is_main:
        print('optimizer train_detector: {}'.format(optimizer_info['train_detector']))
        print('optimizer vit_lr: {} task_lr: {}'.format(optimizer_info['vit_lr'], optimizer_info['task_lr']))
        print(
            'optimizer param groups (vit/task tensors): {}/{}'.format(
                optimizer_info['vit_param_tensors'],
                optimizer_info['task_param_tensors'],
            )
        )
    scheduler = None
    if run_eval:
        scheduler = ReduceLROnPlateau(
            optimizer,
            'max',
            patience=1,
            factor=0.5,
            verbose=True,
            threshold=1e-4,
            threshold_mode='abs',
            min_lr=1e-7,
        )

    start_epoch = _load_resume_checkpoint(
        conf=conf,
        model=model,
        object_detector=object_detector,
        optimizer=optimizer,
        scheduler=scheduler,
        device=gpu_device,
        is_main=is_main,
    )
    if start_epoch >= conf.nepoch and is_main:
        print(
            'resume start_epoch {} >= nepoch {}; no training iterations will run.'.format(
                start_epoch, conf.nepoch
            )
        )

    for epoch in range(start_epoch, conf.nepoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        detector_module = _unwrap(object_detector)
        detector_module.is_train = True
        if train_detector:
            object_detector.train()
        else:
            object_detector.eval()
        start = time.time()
        tr = []

        for b, data in enumerate(dataloader_train):
            if max_train_steps > 0 and b >= max_train_steps:
                break
            im_data = data[0].to(device=gpu_device, non_blocking=True)
            im_info = data[1].to(device=gpu_device, non_blocking=True)
            gt_boxes = data[2].to(device=gpu_device, non_blocking=True)
            num_boxes = data[3].to(device=gpu_device, non_blocking=True)
            gt_annotation = AG_dataset_train.gt_annotations[data[4]]
            if max_video_frames > 0 and im_data.shape[0] > max_video_frames:
                im_data = im_data[:max_video_frames]
                im_info = im_info[:max_video_frames]
                gt_boxes = gt_boxes[:max_video_frames]
                num_boxes = num_boxes[:max_video_frames]
                gt_annotation = gt_annotation[:max_video_frames]

            if train_detector:
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            else:
                # Keep detector frozen for baseline/resnet runs unless explicitly enabled.
                with torch.no_grad():
                    entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred = model(entry)

                attention_distribution = pred['attention_distribution']
                spatial_distribution = pred['spatial_distribution']
                contact_distribution = pred['contacting_distribution']

                attention_label = torch.tensor(pred['attention_gt'], dtype=torch.long).to(device=attention_distribution.device).squeeze()
                if not conf.bce_loss:
                    # multi-label margin loss
                    spatial_label = -torch.ones([len(pred['spatial_gt']), 6], dtype=torch.long).to(device=attention_distribution.device)
                    contact_label = -torch.ones([len(pred['contacting_gt']), 17], dtype=torch.long).to(device=attention_distribution.device)
                    for i in range(len(pred['spatial_gt'])):
                        spatial_label[i, : len(pred['spatial_gt'][i])] = torch.tensor(pred['spatial_gt'][i])
                        contact_label[i, : len(pred['contacting_gt'][i])] = torch.tensor(pred['contacting_gt'][i])
                else:
                    # bce loss
                    spatial_label = torch.zeros([len(pred['spatial_gt']), 6], dtype=torch.float32).to(device=attention_distribution.device)
                    contact_label = torch.zeros([len(pred['contacting_gt']), 17], dtype=torch.float32).to(device=attention_distribution.device)
                    for i in range(len(pred['spatial_gt'])):
                        spatial_label[i, pred['spatial_gt'][i]] = 1
                        contact_label[i, pred['contacting_gt'][i]] = 1

                losses = {}
                if conf.mode == 'sgcls' or conf.mode == 'sgdet':
                    losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

                losses['attention_relation_loss'] = ce_loss(attention_distribution, attention_label)
                if not conf.bce_loss:
                    losses['spatial_relation_loss'] = mlm_loss(spatial_distribution, spatial_label)
                    losses['contact_relation_loss'] = mlm_loss(contact_distribution, contact_label)
                else:
                    losses['spatial_relation_loss'] = bce_loss(spatial_distribution, spatial_label)
                    losses['contact_relation_loss'] = bce_loss(contact_distribution, contact_label)

                loss = sum(losses.values())

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=5, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=5, norm_type=2)
                optimizer.step()

            if is_main:
                tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
                if b % 1000 == 0 and b >= 1000:
                    time_per_batch = (time.time() - start) / 1000
                    print(
                        '\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch'.format(
                            epoch,
                            b,
                            len(dataloader_train),
                            time_per_batch,
                            len(dataloader_train) * time_per_batch / 60,
                        )
                    )
                    mn = pd.concat(tr[-1000:], axis=1).mean(1)
                    print(mn)
                    start = time.time()

        model_to_save = model.module if hasattr(model, 'module') else model
        detector_to_save = _unwrap(object_detector)
        should_save = (epoch % ckpt_every == 0) or (epoch == conf.nepoch - 1)
        if is_main and should_save:
            checkpoint_path = os.path.join(conf.save_path, 'sttran_epoch_{}.pth'.format(epoch))
            checkpoint_payload = {
                'state_dict': model_to_save.state_dict(),
                'model_state_dict': model_to_save.state_dict(),
                'train_detector': bool(train_detector),
                'epoch': epoch,
                'backbone': conf.backbone,
                'det_threshold': conf.det_threshold,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if scheduler is not None:
                checkpoint_payload['scheduler_state_dict'] = scheduler.state_dict()
            if train_detector:
                checkpoint_payload['object_detector_state_dict'] = detector_to_save.state_dict()
            torch.save(checkpoint_payload, checkpoint_path)
            print('*' * 40)
            print('saved checkpoint: {}'.format(checkpoint_path))

        if is_distributed:
            dist.barrier()

        if not run_eval:
            continue

        should_eval = ((epoch + 1) % eval_every == 0) or (epoch == conf.nepoch - 1)
        if not should_eval:
            if is_main:
                print('skip eval at epoch {} (eval_every={})'.format(epoch, eval_every))
            continue

        model.eval()
        detector_module = _unwrap(object_detector)
        detector_module.is_train = False
        object_detector.eval()

        score = 0.0
        if is_main:
            with torch.no_grad():
                for b, data in enumerate(dataloader_test):
                    if max_test_steps > 0 and b >= max_test_steps:
                        break
                    im_data = data[0].to(device=gpu_device, non_blocking=True)
                    im_info = data[1].to(device=gpu_device, non_blocking=True)
                    gt_boxes = data[2].to(device=gpu_device, non_blocking=True)
                    num_boxes = data[3].to(device=gpu_device, non_blocking=True)
                    gt_annotation = AG_dataset_test.gt_annotations[data[4]]

                    entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                    # Keep evaluation outputs in FP32 for NumPy-based evaluator compatibility.
                    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=False):
                        pred = model(entry)
                    evaluator.evaluate_scene_graph(gt_annotation, pred)
                print('-----------', flush=True)
            score = np.mean(evaluator.result_dict[conf.mode + '_recall'][20])
            evaluator.print_stats()
            evaluator.reset_result()

        score_tensor = torch.tensor([score], dtype=torch.float32, device=gpu_device)
        if is_distributed:
            dist.broadcast(score_tensor, src=0)
        if scheduler is not None:
            scheduler.step(float(score_tensor.item()))

        if is_distributed:
            dist.barrier()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()



