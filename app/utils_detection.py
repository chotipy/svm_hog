from typing import List, Tuple
import numpy as np


def nms_xywh(boxes: List[List[float]], scores: List[float], iou_thr: float):
    if not boxes:
        return [], []

    b = np.array(boxes, dtype=float)
    s = np.array(scores, dtype=float)

    x1 = b[:, 0]
    y1 = b[:, 1]
    x2 = b[:, 0] + b[:, 2]
    y2 = b[:, 1] + b[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(s)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]

    return b[keep].tolist(), s[keep].tolist()
