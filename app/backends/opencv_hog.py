import cv2
import numpy as np
from typing import List, Tuple, Optional
from backends.base import BaseDetector


class OpenCVHOGDetector(BaseDetector):
    def __init__(self, config: Optional[dict] = None):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.base_width = 64
        self.base_height = 128
        self.aspect_ratio = self.base_width / self.base_height  # 0.5

        # Default params sesuai notebook (params_strong)
        self.params_strong = {
            "use_preprocessing": True,
            "brightness": 1.25,
            "contrast": 1.15,
            "sharpen": False,
            "win_stride": (4, 4),
            "padding": (8, 8),
            "hit_threshold": 0.20,
            "min_final_score": 0.30,
            "min_person_px": 30,
            "max_person_px": 350,
            "num_scales": 18,
            "min_box_area": 700,
            "max_box_area": None,
            "min_aspect_ratio": 0.30,
            "max_aspect_ratio": 0.70,
            "nms_threshold": 0.35,
        }

        # Default params sesuai notebook (params_weak)
        self.params_weak = {
            "use_preprocessing": True,
            "brightness": 1.25,
            "contrast": 1.15,
            "sharpen": False,
            "win_stride": (4, 4),
            "padding": (8, 8),
            "hit_threshold": 0.10,
            "min_final_score": 0.20,
            "min_person_px": 25,
            "max_person_px": 340,
            "num_scales": 18,
            "min_box_area": 650,
            "max_box_area": None,
            "min_aspect_ratio": 0.28,
            "max_aspect_ratio": 0.72,
            "nms_threshold": 0.38,
        }

        # Update config jika ada input dari luar (misal dari Pickle/Registry)
        if isinstance(config, dict):
            # Jika config memiliki key terpisah untuk strong/weak
            if "params_strong" in config and isinstance(config["params_strong"], dict):
                self.params_strong.update(config["params_strong"])
            if "params_weak" in config and isinstance(config["params_weak"], dict):
                self.params_weak.update(config["params_weak"])

            # Fallback: jika config flat, update keduanya (atau salah satu yang relevan)
            for k, v in config.items():
                if k in self.params_strong:
                    self.params_strong[k] = v
                if k in self.params_weak:
                    self.params_weak[k] = v

    def detect(
        self,
        image_bgr: np.ndarray,
        params: Optional[dict] = None,
        *,
        dual_pass: bool = True,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Main detection method.
        Pastikan image_bgr adalah format BGR (standar OpenCV/cv2.imread).
        """
        # Override params sementara jika ada input params saat runtime
        p_strong = self.params_strong.copy()
        if params:
            p_strong.update(params)

        p_weak = self.params_weak.copy()
        # Jika params punya setting spesifik weak, bisa diupdate disini (opsional)

        if dual_pass:
            return self.dual_pass_detect(image_bgr, p_strong, p_weak)
        else:
            return self._detect_pipeline(image_bgr, p_strong)

    def dual_pass_detect(
        self,
        image_bgr: np.ndarray,
        params_strong: dict,
        params_weak: dict,
    ) -> Tuple[List[List[float]], List[float]]:
        # 1. Detect Strong
        boxes_s, weights_s = self._detect_pipeline(image_bgr, params_strong)

        # 2. Detect Weak
        boxes_w, weights_w = self._detect_pipeline(image_bgr, params_weak)

        # 3. Combine
        all_boxes = boxes_s + boxes_w
        all_weights = weights_s + weights_w

        if not all_boxes:
            return [], []

        # 4. Final NMS menggunakan parameter Strong (sesuai notebook)
        # Notebook: final_boxes, final_weights = detector._improved_nms(all_boxes, all_weights, params_strong)
        nms_boxes, nms_weights = self._improved_nms(
            all_boxes, all_weights, params_strong
        )

        # 5. Final Thresholding menggunakan skor Strong
        final_min = float(params_strong.get("min_final_score", 0.30))
        clean_b, clean_w = [], []

        for b, w in zip(nms_boxes, nms_weights):
            if float(w) >= final_min:
                clean_b.append(b)
                clean_w.append(float(w))

        return clean_b, clean_w

    def _detect_pipeline(
        self, image_bgr: np.ndarray, params: dict
    ) -> Tuple[List[List[float]], List[float]]:
        img = image_bgr.copy()

        # 1. Preprocessing
        if params.get("use_preprocessing", False):
            img = self._preprocess_image(
                img,
                params.get("brightness", 1.25),
                params.get("contrast", 1.15),
                params.get("sharpen", False),
            )

        # 2. Multiscale Detection
        all_boxes, all_weights = self._detect_multiscale_improved(img, params)
        if not all_boxes:
            return [], []

        # 3. Geometric Filtering
        filtered_boxes, filtered_weights = self._filter_boxes(
            all_boxes, all_weights, img.shape, params
        )
        if not filtered_boxes:
            return [], []

        # 4. NMS & Thresholding (Pipeline level)
        nms_boxes, nms_weights = self._improved_nms(
            filtered_boxes, filtered_weights, params
        )

        # Thresholding
        min_score = float(params.get("min_final_score", 0.20))
        final_boxes, final_weights = [], []
        for b, w in zip(nms_boxes, nms_weights):
            if w >= min_score:
                final_boxes.append(b)
                final_weights.append(w)

        return final_boxes, final_weights

    def _detect_multiscale_improved(self, image: np.ndarray, params: dict):
        h, w = image.shape[:2]
        all_boxes: List[List[float]] = []
        all_weights: List[float] = []

        scales = self._calculate_smart_scales(image.shape, params)

        # Pastikan format tuple int untuk OpenCV
        win_stride = tuple(map(int, params.get("win_stride", (4, 4))))
        padding = tuple(map(int, params.get("padding", (8, 8))))
        hit_threshold = float(params.get("hit_threshold", 0.20))

        for scale in scales:
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)

            # Skip jika terlalu kecil
            if scaled_h < 128 or scaled_w < 64:
                continue

            if scale != 1.0:
                scaled_img = cv2.resize(image, (scaled_w, scaled_h))
            else:
                scaled_img = image

            try:
                # DETECT
                boxes, weights = self.hog.detectMultiScale(
                    scaled_img,
                    winStride=win_stride,
                    padding=padding,
                    scale=1.05,
                    hitThreshold=hit_threshold,
                    useMeanshiftGrouping=False,
                )
            except Exception:
                continue

            if len(boxes) > 0:
                # Rescale boxes back to original size
                boxes = boxes.astype(float)
                boxes[:, :2] /= scale
                boxes[:, 2:] /= scale

                all_boxes.extend(boxes.tolist())
                # Flatten weights to 1D list
                all_weights.extend(weights.flatten().tolist())

        return all_boxes, all_weights

    def _calculate_smart_scales(self, img_shape, params: dict):
        h, w = img_shape[:2]
        min_person_height = float(params.get("min_person_px", 30))
        max_person_height = float(params.get("max_person_px", 350))

        min_scale = min_person_height / float(self.base_height)
        max_scale = max_person_height / float(self.base_height)

        # Safety guard
        if min_scale <= 0:
            min_scale = 0.1
        if max_scale <= min_scale:
            max_scale = min_scale + 0.1

        num_scales = int(params.get("num_scales", 18))
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=num_scales)

        return scales.tolist()

    def _filter_boxes(self, boxes, weights, img_shape, params: dict):
        if not boxes:
            return [], []

        boxes = np.array(boxes, dtype=float)
        weights = np.array(weights, dtype=float)

        h, w = img_shape[:2]

        box_widths = boxes[:, 2]
        box_heights = boxes[:, 3]
        box_areas = box_widths * box_heights
        # Hindari division by zero
        aspect_ratios = box_widths / (box_heights + 1e-6)

        min_area = float(params.get("min_box_area", 700))

        max_area = params.get("max_box_area", None)
        if max_area is None:
            max_area = float(w * h) * 0.3
        else:
            max_area = float(max_area)

        min_aspect = float(params.get("min_aspect_ratio", 0.30))
        max_aspect = float(params.get("max_aspect_ratio", 0.70))

        valid_mask = (
            (box_areas >= min_area)
            & (box_areas <= max_area)
            & (aspect_ratios >= min_aspect)
            & (aspect_ratios <= max_aspect)
            & (boxes[:, 0] >= 0)
            & (boxes[:, 1] >= 0)
            & (boxes[:, 0] + boxes[:, 2] <= w)
            & (boxes[:, 1] + boxes[:, 3] <= h)
        )

        return boxes[valid_mask].tolist(), weights[valid_mask].tolist()

    def _improved_nms(self, boxes, weights, params: dict):
        if not boxes:
            return [], []

        boxes = np.array(boxes, dtype=float)
        weights = np.array(weights, dtype=float)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        # Aspect ratio bonus logic (Notebook)
        aspect_ratios = boxes[:, 2] / (boxes[:, 3] + 1e-6)
        aspect_bonus = 1.0 - np.abs(aspect_ratios - self.aspect_ratio)
        adjusted_weights = weights * (1.0 + 0.2 * aspect_bonus)

        order = np.argsort(adjusted_weights)[::-1]
        keep = []
        nms_threshold = float(params.get("nms_threshold", 0.35))

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_overlap = np.maximum(0.0, xx2 - xx1)
            h_overlap = np.maximum(0.0, yy2 - yy1)
            intersection = w_overlap * h_overlap

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]

        return boxes[keep].tolist(), weights[keep].tolist()

    def _preprocess_image(
        self, image: np.ndarray, brightness: float, contrast: float, sharpen: bool
    ):
        processed = image.copy()

        if brightness != 1.0 or contrast != 1.0:
            processed = cv2.convertScaleAbs(
                processed,
                alpha=float(contrast),
                beta=(float(brightness) - 1.0) * 50.0,
            )

        if sharpen:
            kernel = np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
            )
            processed = cv2.filter2D(processed, -1, kernel)

        return processed

    def visualize(
        self, image_bgr: np.ndarray, boxes: List[List[float]], weights: List[float]
    ) -> np.ndarray:
        result = image_bgr.copy()

        for box, weight in zip(boxes, weights):
            x, y, w, h = [int(v) for v in box]
            weight = float(weight)

            if weight > 1.5:
                color = (0, 255, 0)
            elif weight > 0.8:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            label = f"{weight:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x, y - th - 4), (x + tw, y), color, -1)
            cv2.putText(
                result, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        num_people = len(boxes)
        h, w = image_bgr.shape[:2]
        density = num_people / max(((w * h) / 1_000_000), 1e-6)

        if density < 5:
            crowd_level = "Low"
            level_color = (0, 255, 0)
        elif density < 15:
            crowd_level = "Medium"
            level_color = (0, 255, 255)
        elif density < 30:
            crowd_level = "High"
            level_color = (0, 165, 255)
        else:
            crowd_level = "Very High"
            level_color = (0, 0, 255)

        cv2.putText(
            result,
            f"People: {num_people}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            result,
            f"Crowd: {crowd_level}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            level_color,
            2,
        )

        return result

    def estimate_crowd_density(
        self, boxes: List[List[float]], image_shape
    ) -> Tuple[float, str]:
        h, w = image_shape[:2]
        image_area = float(w * h)

        total_person_area = 0.0
        for _, _, bw, bh in boxes:
            total_person_area += float(bw) * float(bh)

        density_ratio = total_person_area / max(image_area, 1.0)

        if density_ratio < 0.02:
            level = "Low"
        elif density_ratio < 0.05:
            level = "Medium"
        elif density_ratio < 0.08:
            level = "High"
        else:
            level = "Very High"

        return density_ratio, level
