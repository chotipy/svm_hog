import cv2
from .base import BaseDetector
from utils_detection import nms_xywh


class OpenCVHOGDetector(BaseDetector):
    def __init__(self, config: dict):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.win_stride = tuple(config.get("win_stride", (4, 4)))
        self.padding = tuple(config.get("padding", (8, 8)))
        self.scale = float(config.get("scale", 1.05))
        self.nms_threshold = float(config.get("nms_threshold", 0.35))

    def detect(self, image_bgr):
        boxes, weights = self.hog.detectMultiScale(
            image_bgr,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )

        boxes = [[x, y, w, h] for (x, y, w, h) in boxes]
        scores = weights.flatten().tolist() if len(weights) else []

        return nms_xywh(boxes, scores, self.nms_threshold)
