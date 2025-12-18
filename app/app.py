import cv2
import streamlit as st
import numpy as np
from PIL import Image
import io
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import os
import pickle
import json
from skimage.feature import hog


class ModelType(Enum):
    STANDARD = "OpenCV HOG"
    CUSTOM = "Custom Trained SVM"


@dataclass
class ModelConfig:
    name: str
    model_path: str
    model_type: str
    default_hit_threshold: float
    default_nms: float
    weak_scale: float
    win_stride: int
    pyramid_scale: float


# Safer model dir (relative to this script, not process cwd)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CUSTOM_MODEL_DIR = os.path.join(BASE_DIR, "models")


MODEL_CONFIGS = {
    ModelType.STANDARD: ModelConfig(
        name="OpenCV HOG",
        model_path="default",
        model_type="opencv",
        default_hit_threshold=0.5,
        default_nms=0.2,
        weak_scale=0.7,
        win_stride=8,
        pyramid_scale=1.05,
    ),
    ModelType.CUSTOM: ModelConfig(
        name="Custom Trained SVM",
        model_path=DEFAULT_CUSTOM_MODEL_DIR,  # was "./models"
        model_type="custom",
        default_hit_threshold=0.3,
        default_nms=0.15,
        weak_scale=0.4,
        win_stride=4,
        pyramid_scale=1.03,
    ),
}


class GlobalDefaults:
    BASE_WIDTH = 64
    BASE_HEIGHT = 128
    PADDING = (8, 8)
    NUM_SCALES = 8
    MIN_SCALE_FACTOR = 0.1
    MIN_ASPECT_RATIO = 0.3
    MAX_ASPECT_RATIO = 0.7
    MAX_AREA_RATIO = 0.4
    HIGH_CONF_THRESHOLD = 1.2
    MEDIUM_CONF_THRESHOLD = 0.7


@dataclass
class PreprocessingParams:
    enabled: bool = True
    brightness: float = 1.0
    contrast: float = 1.0
    blur: int = 0
    sharpen: bool = False


@dataclass
class DetectionParams:
    hit_threshold: float = 0.5
    min_final_score: float = 0.5
    nms_threshold: float = 0.2
    min_person_px: int = 40
    max_person_px: int = 200

    model_type: ModelType = ModelType.STANDARD
    weak_scale: float = 0.7
    win_stride: int = 8
    pyramid_scale: float = 1.05

    def get_min_box_area(self) -> int:
        aspect = GlobalDefaults.BASE_WIDTH / GlobalDefaults.BASE_HEIGHT
        return int((self.min_person_px * aspect * self.min_person_px) * 0.4)


class CustomSVMDetector:
    def __init__(self, model_dir: str):
        model_dir = os.path.abspath(model_dir)
        print(f"📂 Loading custom detector from: {model_dir}")

        classifier_path = os.path.join(model_dir, "body_detector_svm_optimized.pkl")
        scaler_path = os.path.join(model_dir, "feature_scaler_optimized.pkl")
        config_path = os.path.join(model_dir, "config_optimized.json")

        if not all(
            os.path.exists(p) for p in [classifier_path, scaler_path, config_path]
        ):
            raise FileNotFoundError(
                "Model files not found. Expected:\n"
                f"- {classifier_path}\n- {scaler_path}\n- {config_path}"
            )

        with open(classifier_path, "rb") as f:
            self.classifier = pickle.load(f)

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(config_path, "r") as f:
            config = json.load(f)

        self.window_size = tuple(config["window_size"])
        self.hog_params = config["hog_params"]

        params = config["detection_params"]
        self.min_confidence = params["min_confidence"]
        self.nms_threshold = params["nms_threshold"]
        self.step_size = params["step_size"]
        self.max_boxes = params["max_boxes"]
        self.min_box_area = params["min_box_area"]
        self.edge_margin = params["edge_margin"]
        self.min_aspect_ratio = params["min_aspect_ratio"]
        self.max_aspect_ratio = params["max_aspect_ratio"]

        print(
            f"✅ Custom SVM loaded successfully (v{config.get('version', 'unknown')})"
        )

    def detect_multiscale(
        self, image: np.ndarray
    ) -> Tuple[List[List[float]], List[float]]:
        scales = [0.7, 0.85, 1.0, 1.2, 1.4]
        all_boxes = []
        all_confidences = []

        for scale in scales:
            boxes, confs = self._detect_single_scale(image, scale)
            all_boxes.extend(boxes)
            all_confidences.extend(confs)

        if not all_boxes:
            return [], []

        filtered_boxes = []
        filtered_confs = []

        h, w = image.shape[:2]
        for box, conf in zip(all_boxes, all_confidences):
            x, y, bw, bh = box
            if bh == 0:
                continue

            aspect = bw / bh
            if not (self.min_aspect_ratio <= aspect <= self.max_aspect_ratio):
                continue

            if bw * bh < self.min_box_area:
                continue

            cx, cy = x + bw // 2, y + bh // 2
            if not (
                self.edge_margin < cx < w - self.edge_margin
                and self.edge_margin < cy < h - self.edge_margin
            ):
                continue

            filtered_boxes.append(box)
            filtered_confs.append(conf)

        if not filtered_boxes:
            return [], []

        final_boxes, final_confs = self._nms(filtered_boxes, filtered_confs)

        if len(final_boxes) > self.max_boxes:
            sorted_idx = np.argsort(final_confs)[::-1][: self.max_boxes]
            final_boxes = [final_boxes[i] for i in sorted_idx]
            final_confs = [final_confs[i] for i in sorted_idx]

        return final_boxes, final_confs

    def _detect_single_scale(
        self, image: np.ndarray, scale: float
    ) -> Tuple[List[List[int]], List[float]]:
        h, w = image.shape[:2]
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))

        boxes = []
        confidences = []

        # +1 so we include last valid window
        for y in range(0, resized.shape[0] - self.window_size[1] + 1, self.step_size):
            for x in range(
                0, resized.shape[1] - self.window_size[0] + 1, self.step_size
            ):
                window = resized[
                    y : y + self.window_size[1], x : x + self.window_size[0]
                ]

                if (
                    window.shape[0] != self.window_size[1]
                    or window.shape[1] != self.window_size[0]
                ):
                    continue

                features = self._extract_hog(window)
                features = self.scaler.transform([features])

                pred = self.classifier.predict(features)[0]
                conf = self.classifier.decision_function(features)[0]

                if pred == 1 and conf >= self.min_confidence:
                    x_orig = int(x / scale)
                    y_orig = int(y / scale)
                    w_orig = int(self.window_size[0] / scale)
                    h_orig = int(self.window_size[1] / scale)

                    boxes.append([x_orig, y_orig, w_orig, h_orig])
                    confidences.append(float(conf))

        return boxes, confidences

    def _extract_hog(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = hog(
            image,
            orientations=self.hog_params["orientations"],
            pixels_per_cell=tuple(self.hog_params["pixels_per_cell"]),
            cells_per_block=tuple(self.hog_params["cells_per_block"]),
            block_norm=self.hog_params["block_norm"],
            transform_sqrt=self.hog_params["transform_sqrt"],
            feature_vector=True,
        )
        return features

    def _nms(
        self, boxes: List[List[int]], confidences: List[float]
    ) -> Tuple[List[List[int]], List[float]]:
        if not boxes:
            return [], []

        boxes = np.array(boxes)
        confidences = np.array(confidences)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        idxs = np.argsort(confidences)[::-1]
        keep = []

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / areas[idxs[1:]]
            idxs = np.delete(
                idxs,
                np.concatenate(([0], np.where(overlap > self.nms_threshold)[0] + 1)),
            )

        return boxes[keep].tolist(), confidences[keep].tolist()


class UnifiedHOGDetector:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.model_type = model_config.model_type

        if self.model_type == "opencv":
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.custom_detector = None
            print("✅ OpenCV HOG detector loaded")

        elif self.model_type == "custom":
            try:
                self.custom_detector = CustomSVMDetector(model_config.model_path)
                self.hog = None
            except Exception as e:
                print(f"⚠️ Failed to load custom model: {e}")
                print("  Falling back to OpenCV HOG...")
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.custom_detector = None
                self.model_type = "opencv"

    def detect_triple_pass(
        self,
        image_bgr: np.ndarray,
        detection_params: DetectionParams,
        preprocessing_params: PreprocessingParams,
    ) -> Tuple[List[List[float]], List[float]]:
        if image_bgr is None or image_bgr.size == 0:
            return [], []

        processed_img = self._preprocess(image_bgr, preprocessing_params)

        if self.model_type == "custom" and self.custom_detector:
            return self.custom_detector.detect_multiscale(processed_img)

        all_boxes = []
        all_weights = []

        boxes_strong, weights_strong = self._detect_pass_opencv(
            processed_img, detection_params, scale_factor=1.0
        )
        all_boxes.extend(boxes_strong)
        all_weights.extend(weights_strong)

        boxes_medium, weights_medium = self._detect_pass_opencv(
            processed_img, detection_params, scale_factor=0.75
        )
        all_boxes.extend(boxes_medium)
        all_weights.extend(weights_medium)

        boxes_weak, weights_weak = self._detect_pass_opencv(
            processed_img, detection_params, scale_factor=detection_params.weak_scale
        )
        all_boxes.extend(boxes_weak)
        all_weights.extend(weights_weak)

        if not all_boxes:
            return [], []

        nms_boxes, nms_weights = self._apply_nms(
            all_boxes, all_weights, detection_params.nms_threshold
        )
        return self._apply_score_threshold(
            nms_boxes, nms_weights, detection_params.min_final_score
        )

    def _preprocess(
        self, image_bgr: np.ndarray, params: PreprocessingParams
    ) -> np.ndarray:
        if not params.enabled:
            return image_bgr

        processed = image_bgr.copy()

        if params.brightness != 1.0 or params.contrast != 1.0:
            processed = cv2.convertScaleAbs(
                processed,
                alpha=params.contrast,
                beta=int((params.brightness - 1.0) * 50),
            )

        if params.blur > 0:
            kernel_size = params.blur
            if kernel_size % 2 == 0:
                kernel_size += 1
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)

        if params.sharpen:
            kernel = np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
            )
            processed = cv2.filter2D(processed, -1, kernel)

        return processed

    def _detect_pass_opencv(
        self, image_bgr: np.ndarray, params: DetectionParams, scale_factor: float
    ) -> Tuple[List[List[float]], List[float]]:
        h, w = image_bgr.shape[:2]
        all_boxes = []
        all_weights = []

        hit_threshold = params.hit_threshold * scale_factor
        scales = self._calculate_scales(
            image_bgr.shape, params.min_person_px, params.max_person_px
        )

        for scale in scales:
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)

            if (
                scaled_h < GlobalDefaults.BASE_HEIGHT
                or scaled_w < GlobalDefaults.BASE_WIDTH
            ):
                continue

            if scale != 1.0:
                scaled_img = cv2.resize(
                    image_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
                )
            else:
                scaled_img = image_bgr

            try:
                boxes, weights = self.hog.detectMultiScale(
                    scaled_img,
                    winStride=(params.win_stride, params.win_stride),
                    padding=GlobalDefaults.PADDING,
                    scale=params.pyramid_scale,
                    hitThreshold=hit_threshold,
                    useMeanshiftGrouping=False,
                    finalThreshold=0,
                )

                if len(boxes) > 0:
                    boxes = boxes.astype(float)
                    boxes[:, 0] /= scale
                    boxes[:, 1] /= scale
                    boxes[:, 2] /= scale
                    boxes[:, 3] /= scale

                    all_boxes.extend(boxes.tolist())
                    all_weights.extend(weights.flatten().tolist())

            except cv2.error:
                continue

        return self._filter_boxes(all_boxes, all_weights, image_bgr.shape, params)

    def _calculate_scales(
        self, img_shape: Tuple[int, ...], min_person_px: int, max_person_px: int
    ) -> List[float]:
        min_scale = max(
            min_person_px / GlobalDefaults.BASE_HEIGHT, GlobalDefaults.MIN_SCALE_FACTOR
        )
        max_scale = max(max_person_px / GlobalDefaults.BASE_HEIGHT, min_scale + 0.1)

        scales = np.logspace(
            np.log10(min_scale), np.log10(max_scale), num=GlobalDefaults.NUM_SCALES
        )
        return scales.tolist()

    def _filter_boxes(
        self,
        boxes: List[List[float]],
        weights: List[float],
        img_shape: Tuple[int, ...],
        params: DetectionParams,
    ) -> Tuple[List[List[float]], List[float]]:
        if not boxes:
            return [], []

        boxes_arr = np.array(boxes, dtype=float)
        weights_arr = np.array(weights, dtype=float)
        h, w = img_shape[:2]

        widths = boxes_arr[:, 2]
        heights = boxes_arr[:, 3]
        areas = widths * heights
        aspect_ratios = widths / (heights + 1e-6)

        min_area = params.get_min_box_area()
        max_area = w * h * GlobalDefaults.MAX_AREA_RATIO

        valid_mask = (
            (areas >= min_area)
            & (areas <= max_area)
            & (aspect_ratios >= GlobalDefaults.MIN_ASPECT_RATIO)
            & (aspect_ratios <= GlobalDefaults.MAX_ASPECT_RATIO)
            & (boxes_arr[:, 0] >= 0)
            & (boxes_arr[:, 1] >= 0)
            & (boxes_arr[:, 0] + boxes_arr[:, 2] <= w)
            & (boxes_arr[:, 1] + boxes_arr[:, 3] <= h)
        )

        return boxes_arr[valid_mask].tolist(), weights_arr[valid_mask].tolist()

    def _apply_nms(
        self, boxes: List[List[float]], weights: List[float], nms_threshold: float
    ) -> Tuple[List[List[float]], List[float]]:
        if not boxes:
            return [], []

        boxes_arr = np.array(boxes, dtype=float)
        weights_arr = np.array(weights, dtype=float)

        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = x1 + boxes_arr[:, 2]
        y2 = y1 + boxes_arr[:, 3]
        areas = boxes_arr[:, 2] * boxes_arr[:, 3]

        order = np.argsort(weights_arr)[::-1]
        keep_indices = []

        while len(order) > 0:
            i = order[0]
            keep_indices.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_overlap = np.maximum(0, xx2 - xx1)
            h_overlap = np.maximum(0, yy2 - yy1)
            intersection = w_overlap * h_overlap

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)

            order = order[1:][iou <= nms_threshold]

        return boxes_arr[keep_indices].tolist(), weights_arr[keep_indices].tolist()

    def _apply_score_threshold(
        self, boxes: List[List[float]], weights: List[float], min_score: float
    ) -> Tuple[List[List[float]], List[float]]:
        if not boxes:
            return [], []
        filtered_boxes, filtered_weights = [], []
        for box, weight in zip(boxes, weights):
            if weight >= min_score:
                filtered_boxes.append(box)
                filtered_weights.append(weight)
        return filtered_boxes, filtered_weights

    def visualize(
        self,
        image_bgr: np.ndarray,
        boxes: List[List[float]],
        weights: List[float],
        stats: Dict[str, any],
        theme: str = "dark",
    ) -> np.ndarray:
        result = image_bgr.copy()

        if theme == "dark":
            text_color = (255, 255, 255)
            text_bg = (30, 30, 30)
        else:
            text_color = (0, 0, 0)
            text_bg = (240, 240, 240)

        for box, weight in zip(boxes, weights):
            x, y, w, h = [int(v) for v in box]

            if weight > GlobalDefaults.HIGH_CONF_THRESHOLD:
                color = (0, 255, 0)
            elif weight > GlobalDefaults.MEDIUM_CONF_THRESHOLD:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            label = f"{weight:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(
                result, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        y_offset = 30
        for key, value in stats.items():
            text = f"{key}: {value}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(
                result,
                text,
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
            )
            y_offset += 35

        return result


def estimate_crowd_density(
    boxes: List[List[float]], image_shape: Tuple[int, ...]
) -> Tuple[float, str]:
    h, w = image_shape[:2]
    image_area = float(w * h)
    total_person_area = sum(float(bw * bh) for _, _, bw, bh in boxes)
    density_ratio = total_person_area / (image_area + 1e-9)

    if density_ratio < 0.02:
        level = "Low"
    elif density_ratio < 0.05:
        level = "Medium"
    elif density_ratio < 0.08:
        level = "High"
    else:
        level = "Very High"

    return density_ratio, level


def apply_theme(theme: str):
    if theme == "Dark":
        st.markdown(
            """
        <style>
        :root {
            --bg-main: #1f1624;
            --bg-soft: #2a1d30;
            --sidebar: #140d18;
            
            --text-primary: #faf3f7;
            --text-secondary: #f3e7ee;
            
            --accent: #d78ab6;
            --accent-strong: #c45aa3;
            
            --border: rgba(215, 138, 182, 0.35);
        }

        .stApp {
            background:
                radial-gradient(circle at top right,
                    rgba(215,138,182,0.25),
                    transparent 30%),
                radial-gradient(circle at bottom left,
                    rgba(175,66,174,0.18),
                    transparent 50%),
                linear-gradient(135deg, var(--bg-main), var(--bg-soft));
            color: var(--text-primary);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--sidebar), #1b1220);
            border-right: 1px solid var(--border);
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 800;
            text-shadow: 0 1px 6px rgba(215,138,182,0.25);
        }

        .stMarkdown, label {
            color: var(--text-primary) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg,
                var(--accent-strong),
                var(--accent));
            color: #ffffff;
            border-radius: 14px;
            font-weight: 700;
            box-shadow: 0 8px 26px rgba(215,138,182,0.45);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(215,138,182,0.65);
        }

        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #2b1d33, #3a2442);
            border-left: 5px solid var(--accent);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label,
        [data-testid="stSidebar"] [role="radio"] span {
            opacity: 1 !important;
            color: var(--text-primary) !important;
        }
        
        div[data-testid="metric-container"] label {
            color: var(--text-primary) !important;
            font-weight: 700;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #f3e7ee !important;
            font-size: 2rem;
            font-weight: 800;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <style>
        :root {
            --bg-main: #faf3f7;
            --bg-soft: #f3e7ee;
            --sidebar: #f6dde9;

            --text-primary: #3b1f2b;
            --text-secondary: #7a4a63;

            --accent: #c45aa3;
            --accent-soft: #d78ab6;

            --border: rgba(196,90,163,0.25);
        }

        .stApp {
            background: linear-gradient(135deg, var(--bg-main), var(--bg-soft));
            color: var(--text-primary) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--sidebar), #f1d4e3) !important;
            border-right: 1px solid var(--border);
        }

        h1, h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 800;
        }

        .stMarkdown, label {
            color: var(--text-secondary) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg,
                var(--accent),
                var(--accent-soft));
            color: white;
            border-radius: 14px;
            font-weight: 700;
            box-shadow: 0 6px 18px rgba(196,90,163,0.35);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 26px rgba(196,90,163,0.55);
        }

        div[data-testid="metric-container"] {
            background: white;
            border-left: 5px solid var(--accent);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 14px rgba(196,90,163,0.18);
        }

        div[data-testid="metric-container"] label {
            color: var(--accent) !important;
            font-weight: 700;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="Crowd Detector", layout="wide", initial_sidebar_state="expanded"
    )

    st.title("Crowd Detector")
    st.markdown("**Multi-Model Support -> OpenCV and Trained SVM + HOG**")

    st.sidebar.header("Configuration")

    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)
    apply_theme(theme)

    st.sidebar.subheader("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Detection Model",
        options=[m.value for m in ModelType],
        index=0,
    )
    selected_model = ModelType(model_choice)
    model_config = MODEL_CONFIGS[selected_model]

    # ---- Safe session_state init (prevents AttributeError) ----
    if "detector" not in st.session_state:
        st.session_state["detector"] = None
    if "current_model" not in st.session_state:
        st.session_state["current_model"] = None

    # Initialize or switch detector
    if (
        st.session_state["detector"] is None
        or st.session_state["current_model"] != selected_model
    ):
        try:
            st.session_state["detector"] = UnifiedHOGDetector(model_config)
            st.session_state["current_model"] = selected_model
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load model: {str(e)}")
            return

    detector = st.session_state["detector"]

    st.sidebar.write("Selected model:", selected_model)
    st.sidebar.write("Detector loaded:", detector.custom_detector is not None)

    with st.sidebar.expander("Tips on Choosing Model"):
        st.markdown(
            """
            ### OpenCV HOG Mode
            Full control over detection parameters.

            ### Train SVM Mode
            Fixed pre-trained model for dense scenes (UI tuning disabled).
            """
        )

    # Preprocessing
    st.sidebar.subheader("Preprocessing")
    preprocessing_enabled = st.sidebar.checkbox("Enable", False)

    if preprocessing_enabled:
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        blur_val = st.sidebar.select_slider("Blur", options=[0, 3, 5, 7], value=0)
        sharpen = st.sidebar.checkbox("Sharpen", False)
    else:
        brightness, contrast, blur_val, sharpen = 1.0, 1.0, 0, False

    is_custom = selected_model == ModelType.CUSTOM

    # Detection Parameters
    st.sidebar.subheader("Customize Your Detection")
    st.sidebar.subheader("Scale Range")

    # These sliders do NOT affect your current custom detector; disable to avoid confusion
    min_person_px = st.sidebar.slider(
        "Min Height (px)", 25, 100, 40, 5, disabled=is_custom
    )
    max_person_px = st.sidebar.slider(
        "Max Height (px)", 100, 400, 200, 10, disabled=is_custom
    )

    hit_threshold = st.sidebar.slider(
        "Hit Threshold",
        0.0,
        1.0,
        model_config.default_hit_threshold,
        0.05,
        disabled=is_custom,
        help="OpenCV HOG only",
    )

    nms_threshold = st.sidebar.slider(
        "NMS Threshold",
        0.05,
        0.4,
        model_config.default_nms,
        0.01,
        disabled=is_custom,
        help="OpenCV HOG only (custom uses its own NMS threshold from config)",
    )

    min_final_score = st.sidebar.slider(
        "Min Final Confidence",
        0.0,
        1.0,
        model_config.default_hit_threshold,
        0.05,
        disabled=is_custom,
        help="OpenCV HOG only",
    )

    if is_custom:
        st.sidebar.info(
            "Custom SVM mode: tuning sliders are locked (uses config_optimized.json)."
        )

    uploaded_file = st.file_uploader("Upload Your Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("Upload your image to start the detection")
        return

    pil_img = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_img)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    preprocessing_params = PreprocessingParams(
        enabled=preprocessing_enabled,
        brightness=brightness,
        contrast=contrast,
        blur=blur_val,
        sharpen=sharpen,
    )

    detection_params = DetectionParams(
        hit_threshold=hit_threshold,
        min_final_score=min_final_score,
        nms_threshold=nms_threshold,
        min_person_px=min_person_px,
        max_person_px=max_person_px,
        model_type=selected_model,
        weak_scale=model_config.weak_scale,
        win_stride=model_config.win_stride,
        pyramid_scale=model_config.pyramid_scale,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Picture")
        st.image(pil_img, use_container_width=True)

    with st.spinner("Detecting..."):
        boxes, weights = detector.detect_triple_pass(
            image_bgr, detection_params, preprocessing_params
        )

        density_ratio, crowd_level = estimate_crowd_density(boxes, image_bgr.shape)
        people_count = len(boxes)

        stats = {
            "People": people_count,
            "Crowd": crowd_level,
            "Density": f"{density_ratio:.3f}",
        }

        result_bgr = detector.visualize(image_bgr, boxes, weights, stats, theme.lower())
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Processed Picture")
        st.image(result_rgb, use_container_width=True)

    st.markdown("---")
    col3, col4, col5, col6 = st.columns(4)

    avg_conf = float(np.mean(weights)) if weights else 0.0
    emoji_map = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Very High": "🔴"}

    with col3:
        st.metric("People", people_count)
    with col4:
        st.metric("Crowd", f"{emoji_map.get(crowd_level, '')} {crowd_level}")
    with col5:
        st.metric("Density", f"{density_ratio:.3f}")
    with col6:
        st.metric("Confidence", f"{avg_conf:.2f}")

    if boxes:
        with st.expander("📋 Detections"):
            for i, (box, weight) in enumerate(zip(boxes, weights)):
                x, y, w, h = [int(v) for v in box]
                st.text(f"#{i+1}: ({x},{y}) {w}x{h} conf={weight:.2f}")

    st.markdown("---")
    buf = io.BytesIO()
    Image.fromarray(result_rgb).save(buf, format="PNG")
    st.download_button(
        "⬇️ Download",
        buf.getvalue(),
        f"detection_{model_config.name.lower().replace(' ', '_')}.png",
        "image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
