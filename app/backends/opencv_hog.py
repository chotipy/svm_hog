import cv2


class OpenCVHOGDetector:
    def __init__(self, config):
        self.hog = config["hog"]
        self.winStride = tuple(config["winStride"])
        self.padding = tuple(config["padding"])
        self.scale = float(config["scale"])

    def detect(self, image_bgr):
        boxes, weights = self.hog.detectMultiScale(
            image_bgr,
            winStride=self.winStride,
            padding=self.padding,
            scale=self.scale,
        )
        boxes = boxes.tolist() if len(boxes) else []
        scores = weights.flatten().tolist() if len(weights) else []
        return boxes, scores
