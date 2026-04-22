import importlib
import os
import subprocess
import sys

os.environ.setdefault('SGCLS_DUPLICATE_POLICY', 'iou')
os.environ.setdefault('SGCLS_LABEL_SOURCE', 'detector')
os.environ.setdefault('PHASE3_ASSERT_DINOV2_PATH', '1')

phase3_ag = importlib.import_module('dataloader.action_genome_phase3')
phase3_detector_data = importlib.import_module('dataloader.action_genome_detector_phase3')
phase3_detector = importlib.import_module('lib.object_detector_phase3')

sys.modules['dataloader.action_genome'] = phase3_ag
sys.modules['dataloader.action_genome_detector'] = phase3_detector_data
sys.modules['lib.object_detector'] = phase3_detector

base = importlib.import_module('train_detector_stage1')


def _run_detector_map_eval(args, checkpoint_path, summary_path):
    cmd = [
        sys.executable,
        '-u',
        'evaluate_detector_map_phase3.py',
        '-data_path',
        args.data_path,
        '-model_path',
        checkpoint_path,
        '-datasize',
        args.datasize,
        '--backbone',
        'dinov2',
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
    print('running phase3 detector mAP eval:')
    print('  {}'.format(' '.join(cmd)))
    return subprocess.run(cmd, check=False)


base._run_detector_map_eval = _run_detector_map_eval


if __name__ == '__main__':
    base.main()
