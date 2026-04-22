import importlib
import os
import subprocess
import sys

os.environ.setdefault('SGCLS_DUPLICATE_POLICY', 'iou')
os.environ.setdefault('SGCLS_LABEL_SOURCE', 'detector')
os.environ.setdefault('PHASE3_ASSERT_DINOV2_PATH', '1')

phase3_ag = importlib.import_module('dataloader.action_genome_phase3')
phase3_detector = importlib.import_module('lib.object_detector_phase3')

sys.modules['dataloader.action_genome'] = phase3_ag
sys.modules['lib.object_detector'] = phase3_detector

base = importlib.import_module('train')


def _run_detector_map_eval(conf, checkpoint_path, max_steps, max_video_frames, num_workers, iou_threshold):
    if not os.path.isfile(checkpoint_path):
        print('phase3 detector mAP eval skipped: checkpoint missing -> {}'.format(checkpoint_path))
        return 1
    cmd = [
        sys.executable,
        '-u',
        'evaluate_detector_map_phase3.py',
        '-model_path',
        checkpoint_path,
        '-data_path',
        conf.data_path,
        '-datasize',
        conf.datasize,
        '--backbone',
        'dinov2',
        '--det_threshold',
        str(conf.det_threshold),
        '--max_steps',
        str(max_steps),
        '--max_video_frames',
        str(max_video_frames),
        '--num_workers',
        str(num_workers),
        '--iou_threshold',
        str(iou_threshold),
    ]
    print('running phase3 detector mAP eval command:\n  {}'.format(' '.join(cmd)))
    result = subprocess.run(cmd, check=False)
    print('phase3 detector mAP eval exit code:', result.returncode)
    return result.returncode


base._run_detector_map_eval = _run_detector_map_eval


if __name__ == '__main__':
    base.main()
