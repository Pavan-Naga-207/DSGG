# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torchvision.ops import nms as tv_nms


def nms(boxes, scores, iou_threshold):
    # Use torchvision's maintained NMS op to avoid custom extension build issues.
    return tv_nms(boxes, scores, iou_threshold)
