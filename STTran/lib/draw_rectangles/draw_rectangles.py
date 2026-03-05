import numpy as np


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, 0.0), 1.0)


def draw_union_boxes(bbox_pairs, pooling_size, padding=0):
    """
    Python/Numpy fallback for Cython draw_union_boxes.

    Args:
        bbox_pairs: [N, 8] as [x1,y1,x2,y2,x1,y1,x2,y2].
        pooling_size: output mask size.
        padding: only 0 is supported (same as original Cython code).
    Returns:
        [N, 2, pooling_size, pooling_size] float32 masks.
    """
    if padding != 0:
        raise AssertionError("Padding>0 not supported yet")

    box_pairs = np.asarray(bbox_pairs, dtype=np.float32)
    n_pairs = box_pairs.shape[0]
    out = np.zeros((n_pairs, 2, pooling_size, pooling_size), dtype=np.float32)

    xs = np.arange(pooling_size, dtype=np.float32)
    ys = np.arange(pooling_size, dtype=np.float32)

    for n in range(n_pairs):
        x1_union = min(box_pairs[n, 0], box_pairs[n, 4])
        y1_union = min(box_pairs[n, 1], box_pairs[n, 5])
        x2_union = max(box_pairs[n, 2], box_pairs[n, 6])
        y2_union = max(box_pairs[n, 3], box_pairs[n, 7])

        w = max(x2_union - x1_union, 1e-6)
        h = max(y2_union - y1_union, 1e-6)

        for i in range(2):
            b = 4 * i
            x1_box = (box_pairs[n, b + 0] - x1_union) * pooling_size / w
            y1_box = (box_pairs[n, b + 1] - y1_union) * pooling_size / h
            x2_box = (box_pairs[n, b + 2] - x1_union) * pooling_size / w
            y2_box = (box_pairs[n, b + 3] - y1_union) * pooling_size / h

            y_contrib = _clip01(ys + 1 - y1_box) * _clip01(y2_box - ys)
            x_contrib = _clip01(xs + 1 - x1_box) * _clip01(x2_box - xs)
            out[n, i] = np.outer(y_contrib, x_contrib)

    return out
