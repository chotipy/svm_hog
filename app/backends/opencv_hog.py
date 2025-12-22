import cv2
import os
import random
import numpy as np
from PIL import Image
import io
from typing import List, Tuple
from .base import BaseDetector


class OpenCVHOGDetector(BaseDetector):
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.base_w = 64
        self.base_h = 128
        self.aspect_ratio = self.base_w / self.base_h  # 0.5

    def detect(self, image_bgr, params):
        if params.get("use_preprocessing", False):
            image_bgr = self._preprocess(
                image_bgr,
                params.get("brightness", 1.25),
                params.get("contrast", 1.15),
                params.get("sharpen", False),
            )

        boxes, scores = self._detect_multiscale(image_bgr, params)
        if not boxes:
            return [], []

        boxes, scores = self._filter_boxes(boxes, scores, image_bgr.shape, params)
        if not boxes:
            return [], []

        boxes, scores = self._nms(boxes, scores, params)

        min_score = params.get("min_final_score", 0.3)
        final_b, final_s = [], []
        for b, s in zip(boxes, scores):
            if s >= min_score:
                final_b.append(b)
                final_s.append(s)

        return final_b, final_s

    def _detect_multiscale(self, image, params):
        h, w = image.shape[:2]
        boxes_all, scores_all = [], []

        scales = self._smart_scales(image.shape, params)

        for s in scales:
            sh, sw = int(h * s), int(w * s)
            if sh < 128 or sw < 64:
                continue

            img_s = cv2.resize(image, (sw, sh)) if s != 1.0 else image

            try:
                boxes, scores = self.hog.detectMultiScale(
                    img_s,
                    winStride=params.get("win_stride", (4, 4)),
                    padding=params.get("padding", (8, 8)),
                    scale=1.05,
                    hitThreshold=params.get("hit_threshold", 0.2),
                )
            except Exception:
                continue

            if len(boxes) == 0:
                continue

            boxes = boxes.astype(float)
            boxes[:, :2] /= s
            boxes[:, 2:] /= s

            boxes_all.extend(boxes.tolist())
            scores_all.extend(scores.flatten().tolist())

        return boxes_all, scores_all

    def _smart_scales(self, shape, params):
        min_h = params.get("min_person_px", 30)
        max_h = params.get("max_person_px", 350)

        min_s = min_h / self.base_h
        max_s = max_h / self.base_h

        n = params.get("num_scales", 18)
        return np.logspace(np.log10(min_s), np.log10(max_s), n).tolist()

    def _filter_boxes(self, boxes, scores, shape, params):
        boxes = np.array(boxes)
        scores = np.array(scores)

        h, w = shape[:2]
        bw, bh = boxes[:, 2], boxes[:, 3]
        area = bw * bh
        aspect = bw / (bh + 1e-6)

        mask = (
            (area >= params.get("min_box_area", 700))
            & (aspect >= params.get("min_aspect_ratio", 0.3))
            & (aspect <= params.get("max_aspect_ratio", 0.7))
            & (boxes[:, 0] >= 0)
            & (boxes[:, 1] >= 0)
            & (boxes[:, 0] + bw <= w)
            & (boxes[:, 1] + bh <= h)
        )

        return boxes[mask].tolist(), scores[mask].tolist()

    def _nms(self, boxes, scores, params):
        boxes = np.array(boxes)
        scores = np.array(scores)

        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        aspect = boxes[:, 2] / (boxes[:, 3] + 1e-6)
        bonus = 1.0 - np.abs(aspect - self.aspect_ratio)
        scores = scores * (1.0 + 0.2 * bonus)

        order = scores.argsort()[::-1]
        keep = []

        thr = params.get("nms_threshold", 0.35)

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= thr]

        return boxes[keep].tolist(), scores[keep].tolist()

    def _preprocess(self, img, brightness, contrast, sharpen):
        out = cv2.convertScaleAbs(
            img,
            alpha=contrast,
            beta=(brightness - 1.0) * 50,
        )
        if sharpen:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            out = cv2.filter2D(out, -1, kernel)
        return out


params_strong = {
    "use_preprocessing": True,
    "brightness": 1.25,  # Slightly reduced for better quality
    "contrast": 1.15,  # Increased contrast helps
    "sharpen": False,
    "win_stride": (4, 4),
    "padding": (8, 8),
    "hit_threshold": 0.20,  # Higher threshold for better quality
    "min_final_score": 0.30,  # Balanced threshold
    "min_person_px": 30,  # Back to detecting more visible people
    "max_person_px": 350,
    "num_scales": 18,  # More scales for better coverage
    "min_box_area": 700,  # Higher to avoid small false positives
    "max_box_area": None,
    "min_aspect_ratio": 0.30,  # Stricter aspect ratio (closer to 0.5 ideal)
    "max_aspect_ratio": 0.70,  # People should be roughly this shape
    "nms_threshold": 0.35,  # Stricter NMS to remove duplicates
}

params_weak = {
    "use_preprocessing": True,
    "brightness": 1.25,
    "contrast": 1.15,
    "sharpen": False,
    "win_stride": (4, 4),
    "padding": (8, 8),
    "hit_threshold": 0.10,  # Moderate threshold
    "min_final_score": 0.20,  # Higher to catch only good detections
    "min_person_px": 25,  # Catch slightly smaller people
    "max_person_px": 340,
    "num_scales": 18,
    "min_box_area": 650,  # Lower than strong pass
    "max_box_area": None,
    "min_aspect_ratio": 0.28,  # Still strict
    "max_aspect_ratio": 0.72,
    "nms_threshold": 0.38,  # Moderate NMS
}


def dual_pass_detect(detector, image, params_strong, params_weak):
    boxes_s, weights_s = detector.detect(image, params_strong)
    boxes_w, weights_w = detector.detect(image, params_weak)

    all_boxes = boxes_s + boxes_w
    all_weights = weights_s + weights_w

    # final NMS pakai strong
    final_boxes, final_weights = detector._improved_nms(
        all_boxes, all_weights, params_strong
    )

    # final threshold pakai weak (lebih recall)
    final_min = params_strong.get("min_final_score")
    clean_b, clean_w = [], []
    for b, w in zip(final_boxes, final_weights):
        if w >= final_min:
            clean_b.append(b)
            clean_w.append(w)

    return clean_b, clean_w


def estimate_crowd_density(boxes, image_shape):
    h, w = image_shape[:2]
    image_area = w * h

    total_person_area = 0
    for _, _, bw, bh in boxes:
        total_person_area += bw * bh

    density_ratio = total_person_area / image_area

    # heuristic mapping
    if density_ratio < 0.02:
        level = "Low"
    elif density_ratio < 0.05:
        level = "Medium"
    elif density_ratio < 0.08:
        level = "High"
    else:
        level = "Very High"

    return density_ratio, level
