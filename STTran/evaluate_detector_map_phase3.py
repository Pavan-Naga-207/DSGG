import importlib
import os
import sys

os.environ.setdefault('SGCLS_DUPLICATE_POLICY', 'iou')
os.environ.setdefault('SGCLS_LABEL_SOURCE', 'detector')
os.environ.setdefault('PHASE3_ASSERT_DINOV2_PATH', '1')

phase3_ag = importlib.import_module('dataloader.action_genome_phase3')
phase3_detector = importlib.import_module('lib.object_detector_phase3')

sys.modules['dataloader.action_genome'] = phase3_ag
sys.modules['lib.object_detector'] = phase3_detector

base = importlib.import_module('evaluate_detector_map')


if __name__ == '__main__':
    base.main()
