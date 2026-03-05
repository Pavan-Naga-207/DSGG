import numpy as np


def _as_float64_boxes(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected shape [N,4], got {arr.shape}")
    return arr


def bbox_overlaps(boxes, query_boxes):
    """
    Python/Numpy fallback for Cython bbox_overlaps.
    Returns IoU matrix of shape [N, K].
    """
    boxes = _as_float64_boxes(boxes)
    query_boxes = _as_float64_boxes(query_boxes)

    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float64)

    xA = np.maximum(boxes[:, None, 0], query_boxes[None, :, 0])
    yA = np.maximum(boxes[:, None, 1], query_boxes[None, :, 1])
    xB = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2])
    yB = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3])

    iw = xB - xA + 1.0
    ih = yB - yA + 1.0
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)

    box_area = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    query_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1.0) * (
        query_boxes[:, 3] - query_boxes[:, 1] + 1.0
    )
    union = box_area[:, None] + query_area[None, :] - inter
    union = np.maximum(union, 1e-12)

    return inter / union


def bbox_intersections(boxes, query_boxes):
    """
    Python/Numpy fallback for Cython bbox_intersections.
    Returns intersection ratio covered by query boxes, shape [N, K].
    """
    boxes = _as_float64_boxes(boxes)
    query_boxes = _as_float64_boxes(query_boxes)

    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float64)

    xA = np.maximum(boxes[:, None, 0], query_boxes[None, :, 0])
    yA = np.maximum(boxes[:, None, 1], query_boxes[None, :, 1])
    xB = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2])
    yB = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3])

    iw = xB - xA + 1.0
    ih = yB - yA + 1.0
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)

    query_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1.0) * (
        query_boxes[:, 3] - query_boxes[:, 1] + 1.0
    )
    query_area = np.maximum(query_area, 1e-12)
    return inter / query_area[None, :]
