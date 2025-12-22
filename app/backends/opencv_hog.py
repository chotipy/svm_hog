import cv2
import numpy as np
from typing import List, Tuple
from .base import BaseDetector


class OpenCVHOGDetector(BaseDetector):
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image_bgr: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        boxes, weights = self.hog.detectMultiScale(
            image_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=0.0,
            useMeanshiftGrouping=False,
        )
        if len(boxes) == 0:
            return [], []
        return boxes.astype(float).tolist(), weights.flatten().astype(float).tolist()
