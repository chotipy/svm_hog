import cv2
import streamlit as st
import numpy as np
from PIL import Image
import io
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* COLOR PALETTE */
        :root {
            --imperial-blue: #03256c;
            --persian-blue: #2541b2;
            --ocean-deep: #1768ac;
            --sky-surge: #06bee1;
            --white: #ffffff;
        }
        
        /* MAIN LAYOUT */
        .stApp {
            background-color: #f4f6f9;
        }
        
        /* SIDEBAR STYLING */
        [data-testid="stSidebar"] {
            background-color: var(--imperial-blue);
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: var(--sky-surge) !important;
            font-family: 'Helvetica Neue', sans-serif;
        }
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] .stMarkdown {
            color: var(--white) !important;
        }
        
        /* WIDGETS */
        div[data-testid="stSlider"] > label {
            color: var(--imperial-blue); 
            font-weight: bold;
        }
        
        /* CUSTOM BUTTONS */
        .stButton>button {
            background-color: var(--persian-blue);
            color: var(--white);
            border: none;
            border-radius: 6px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: var(--ocean-deep);
            color: var(--white);
        }

        /* METRICS CARDS */
        div[data-testid="metric-container"] {
            background-color: var(--white);
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid var(--sky-surge);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


@dataclass
class DetectionParams:
    """Parameters for HOG detection"""

    # Preprocessing
    use_preprocessing: bool = False
    brightness: float = 1.0
    contrast: float = 1.0
    sharpen: bool = False

    # Core detection
    win_stride: Tuple[int, int] = (4, 4)
    padding: Tuple[int, int] = (8, 8)
    hit_threshold: float = 0.6
    min_final_score: float = 0.6

    # Multi-scale
    min_person_px: int = 40
    max_person_px: int = 220
    num_scales: int = 6

    # Filtering
    min_box_area: int = 1500
    max_box_area: Optional[int] = None
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 0.65

    # NMS
    nms_threshold: float = 0.2


class ImprovedHOGDetector:
    """
    Enhanced HOG detector with dual-pass detection and improved filtering

    Key features:
    - Smart multi-scale pyramid
    - Aspect ratio-aware NMS
    - Dual-pass detection (strong + weak)
    - Area-based density estimation
    """

    # Standard HOG person window dimensions
    BASE_WIDTH = 64
    BASE_HEIGHT = 128
    ASPECT_RATIO = BASE_WIDTH / BASE_HEIGHT  # 0.5

    # Confidence thresholds for color coding
    HIGH_CONF_THRESHOLD = 1.5
    MEDIUM_CONF_THRESHOLD = 0.8

    def __init__(self):
        """Initialize HOG descriptor with default people detector"""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        print("✓ Improved HOG detector initialized")
        print(f"  Base window: {self.BASE_WIDTH}×{self.BASE_HEIGHT}")
        print(f"  Target aspect ratio: {self.ASPECT_RATIO:.2f}")

    def detect(
        self, image_bgr: np.ndarray, params: DetectionParams
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Single-pass detection pipeline

        Pipeline: preprocess -> multiscale -> filter -> NMS -> score filter

        Args:
            image_bgr: Input image in BGR format
            params: Detection parameters

        Returns:
            boxes: List of [x, y, w, h]
            weights: Confidence scores
        """
        if image_bgr is None or image_bgr.size == 0:
            return [], []

        # Preprocess
        processed_img = self._preprocess_image(image_bgr, params)

        # Multi-scale detection
        all_boxes, all_weights = self._detect_multiscale(processed_img, params)
        if not all_boxes:
            return [], []

        # Filter by size and aspect ratio
        filtered_boxes, filtered_weights = self._filter_boxes(
            all_boxes, all_weights, processed_img.shape, params
        )
        if not filtered_boxes:
            return [], []

        # Apply NMS
        nms_boxes, nms_weights = self._apply_nms(
            filtered_boxes, filtered_weights, params
        )
        if not nms_boxes:
            return [], []

        # Final confidence threshold
        final_boxes, final_weights = self._apply_score_threshold(
            nms_boxes, nms_weights, params.min_final_score
        )

        return final_boxes, final_weights

    def detect_pipeline(
        self, image_bgr: np.ndarray, params: DetectionParams
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Detection pipeline without final score threshold (for dual-pass)

        Returns boxes after NMS but before final score filtering
        """
        if image_bgr is None or image_bgr.size == 0:
            return [], []

        processed_img = self._preprocess_image(image_bgr, params)
        all_boxes, all_weights = self._detect_multiscale(processed_img, params)

        if not all_boxes:
            return [], []

        filtered_boxes, filtered_weights = self._filter_boxes(
            all_boxes, all_weights, processed_img.shape, params
        )

        if not filtered_boxes:
            return [], []

        return self._apply_nms(filtered_boxes, filtered_weights, params)

    def _preprocess_image(
        self, image_bgr: np.ndarray, params: DetectionParams
    ) -> np.ndarray:
        """
        Apply preprocessing to improve detection

        Args:
            image_bgr: Input BGR image
            params: Parameters containing brightness, contrast, sharpen flags

        Returns:
            Preprocessed image
        """
        if not params.use_preprocessing:
            return image_bgr

        processed = image_bgr.copy()

        # Brightness and contrast adjustment
        if params.brightness != 1.0 or params.contrast != 1.0:
            processed = cv2.convertScaleAbs(
                processed, alpha=params.contrast, beta=(params.brightness - 1.0) * 50
            )

        # Sharpening filter
        if params.sharpen:
            kernel = np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
            )
            processed = cv2.filter2D(processed, -1, kernel)

        return processed

    def _detect_multiscale(
        self, image_bgr: np.ndarray, params: DetectionParams
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Multi-scale detection with smart scale pyramid

        Uses logarithmic scale distribution for efficient coverage
        """
        h, w = image_bgr.shape[:2]
        all_boxes = []
        all_weights = []

        scales = self._calculate_scales(image_bgr.shape, params)

        for scale in scales:
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)

            # Skip if too small
            if scaled_h < self.BASE_HEIGHT or scaled_w < self.BASE_WIDTH:
                continue

            # Resize image
            scaled_img = (
                cv2.resize(image_bgr, (scaled_w, scaled_h))
                if scale != 1.0
                else image_bgr
            )

            try:
                # Detect at this scale
                boxes, weights = self.hog.detectMultiScale(
                    scaled_img,
                    winStride=params.win_stride,
                    padding=params.padding,
                    scale=1.05,  # Fine-grained pyramid within scale
                    hitThreshold=params.hit_threshold,
                    useMeanshiftGrouping=False,
                )

                if len(boxes) > 0:
                    # Transform boxes back to original scale
                    boxes = boxes.astype(float)
                    boxes[:, :2] /= scale  # x, y
                    boxes[:, 2:] /= scale  # w, h

                    all_boxes.extend(boxes.tolist())
                    all_weights.extend(weights.flatten().tolist())

            except cv2.error:
                # Silently skip failed scales
                continue

        return all_boxes, all_weights

    def _calculate_scales(
        self, img_shape: Tuple[int, ...], params: DetectionParams
    ) -> List[float]:
        """
        Calculate smart scale pyramid using logarithmic distribution

        Based on expected person heights in the scene
        """
        min_scale = max(params.min_person_px / self.BASE_HEIGHT, 0.05)
        max_scale = max(params.max_person_px / self.BASE_HEIGHT, min_scale + 1e-6)

        # Logarithmic spacing for better coverage
        scales = np.logspace(
            np.log10(min_scale), np.log10(max_scale), num=params.num_scales
        )

        return scales.tolist()

    def _filter_boxes(
        self,
        boxes: List[List[float]],
        weights: List[float],
        img_shape: Tuple[int, ...],
        params: DetectionParams,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Filter boxes by size, aspect ratio, and boundaries

        Removes unrealistic detections based on:
        - Box area (too small/large)
        - Aspect ratio (non-person-like shapes)
        - Image boundaries (out of bounds)
        """
        if not boxes:
            return [], []

        boxes_arr = np.array(boxes, dtype=float)
        weights_arr = np.array(weights, dtype=float)

        h, w = img_shape[:2]

        # Calculate metrics
        box_widths = boxes_arr[:, 2]
        box_heights = boxes_arr[:, 3]
        areas = box_widths * box_heights
        aspect_ratios = box_widths / (box_heights + 1e-6)

        # Determine max area
        max_area = params.max_box_area if params.max_box_area else (w * h * 0.3)

        # Apply all filters
        valid_mask = (
            (areas >= params.min_box_area)
            & (areas <= max_area)
            & (aspect_ratios >= params.min_aspect_ratio)
            & (aspect_ratios <= params.max_aspect_ratio)
            & (boxes_arr[:, 0] >= 0)
            & (boxes_arr[:, 1] >= 0)
            & (boxes_arr[:, 0] + boxes_arr[:, 2] <= w)
            & (boxes_arr[:, 1] + boxes_arr[:, 3] <= h)
        )

        return boxes_arr[valid_mask].tolist(), weights_arr[valid_mask].tolist()

    def _apply_nms(
        self, boxes: List[List[float]], weights: List[float], params: DetectionParams
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Improved Non-Maximum Suppression with aspect ratio awareness

        Prefers boxes with aspect ratios closer to the standard person ratio
        """
        if not boxes:
            return [], []

        boxes_arr = np.array(boxes, dtype=float)
        weights_arr = np.array(weights, dtype=float)

        # Calculate box coordinates
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = x1 + boxes_arr[:, 2]
        y2 = y1 + boxes_arr[:, 3]
        areas = boxes_arr[:, 2] * boxes_arr[:, 3]

        # Aspect ratio bonus (favor standard person proportions)
        aspect_ratios = boxes_arr[:, 2] / (boxes_arr[:, 3] + 1e-6)
        aspect_bonus = 1.0 - np.abs(aspect_ratios - self.ASPECT_RATIO)
        aspect_bonus = np.clip(aspect_bonus, 0.0, 1.0)

        # Adjust weights with aspect ratio bonus
        adjusted_weights = weights_arr * (1.0 + 0.2 * aspect_bonus)
        order = np.argsort(adjusted_weights)[::-1]

        keep_indices = []

        while len(order) > 0:
            # Keep highest scoring box
            i = order[0]
            keep_indices.append(i)

            if len(order) == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_overlap = np.maximum(0, xx2 - xx1)
            h_overlap = np.maximum(0, yy2 - yy1)
            intersection = w_overlap * h_overlap

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)

            # Keep only boxes with IoU below threshold
            remaining_mask = iou <= params.nms_threshold
            order = order[1:][remaining_mask]

        return boxes_arr[keep_indices].tolist(), weights_arr[keep_indices].tolist()

    def _apply_score_threshold(
        self, boxes: List[List[float]], weights: List[float], min_score: float
    ) -> Tuple[List[List[float]], List[float]]:
        """Apply minimum confidence score threshold"""
        if not boxes:
            return [], []

        filtered_boxes = []
        filtered_weights = []

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
        people_text: Optional[str] = None,
        crowd_text: Optional[str] = None,
        density_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize detection results with color-coded bounding boxes

        Colors:
        - Green: High confidence (>1.5)
        - Yellow: Medium confidence (0.8-1.5)
        - Orange: Low confidence (<0.8)
        """
        result = image_bgr.copy()

        # Draw bounding boxes
        for box, weight in zip(boxes, weights):
            x, y, w, h = [int(v) for v in box]

            # Determine color based on confidence
            if weight > self.HIGH_CONF_THRESHOLD:
                color = (0, 255, 0)  # Green
            elif weight > self.MEDIUM_CONF_THRESHOLD:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange

            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # Draw confidence label
            label = f"{weight:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(result, (x, y - label_h - 4), (x + label_w, y), color, -1)
            cv2.putText(
                result, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # Add text overlays
        y_offset = 30
        for text in [people_text, crowd_text, density_text]:
            if text:
                cv2.putText(
                    result,
                    text,
                    (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                y_offset += 30

        return result


def dual_pass_detect(
    detector: ImprovedHOGDetector,
    image_bgr: np.ndarray,
    params_strong: DetectionParams,
    params_weak: DetectionParams,
) -> Tuple[List[List[float]], List[float]]:
    """
    Dual-pass detection strategy

    Combines strong (high precision) and weak (high recall) detections
    to improve overall performance in challenging scenarios.

    Args:
        detector: HOG detector instance
        image_bgr: Input image
        params_strong: Parameters for high-precision pass
        params_weak: Parameters for high-recall pass

    Returns:
        boxes: Combined detections
        weights: Confidence scores
    """
    # Strong pass (high precision)
    boxes_strong, weights_strong = detector.detect_pipeline(image_bgr, params_strong)

    # Weak pass (high recall)
    boxes_weak, weights_weak = detector.detect_pipeline(image_bgr, params_weak)

    # Combine detections
    all_boxes = boxes_strong + boxes_weak
    all_weights = weights_strong + weights_weak

    if not all_boxes:
        return [], []

    # Apply final NMS with strong parameters
    final_boxes, final_weights = detector._apply_nms(
        all_boxes, all_weights, params_strong
    )

    # Apply strong confidence threshold
    return detector._apply_score_threshold(
        final_boxes, final_weights, params_strong.min_final_score
    )


def estimate_crowd_density(
    boxes: List[List[float]], image_shape: Tuple[int, ...]
) -> Tuple[float, str]:
    """
    Estimate crowd density based on total person area coverage

    More reliable than simple person count for varying camera angles

    Args:
        boxes: List of bounding boxes [x, y, w, h]
        image_shape: Image dimensions (h, w, c)

    Returns:
        density_ratio: Fraction of image covered by people (0-1)
        level: Textual description of crowd level
    """
    h, w = image_shape[:2]
    image_area = float(w * h)

    # Calculate total area covered by detected people
    total_person_area = sum(float(bw * bh) for _, _, bw, bh in boxes)

    density_ratio = total_person_area / (image_area + 1e-9)

    # Map density ratio to crowd level
    if density_ratio < 0.02:
        level = "Low"
    elif density_ratio < 0.05:
        level = "Medium"
    elif density_ratio < 0.08:
        level = "High"
    else:
        level = "Very High"

    return density_ratio, level


# =========================
# Streamlit Application
# =========================


def create_weak_params(
    params_strong: DetectionParams, weak_scale: float
) -> DetectionParams:
    """Create weakened parameters for recall-boosting pass"""
    params_dict = vars(params_strong).copy()

    # Reduce thresholds for higher recall
    params_dict["hit_threshold"] = max(0.0, params_strong.hit_threshold * weak_scale)
    params_dict["min_final_score"] = max(
        0.0, params_strong.min_final_score * weak_scale
    )
    params_dict["min_person_px"] = max(20, params_strong.min_person_px - 5)
    params_dict["min_box_area"] = max(200, params_strong.min_box_area - 400)

    return DetectionParams(**params_dict)


def main():
    st.set_page_config(page_title="Improved HOG Crowd Detector", layout="wide")

    st.title("🚶‍♂️ Improved Multi-Scale HOG Crowd Detector")
    st.markdown(
        """
    **Dual-Pass Detection with Area-Based Density Estimation**
    
    Based on Dalal & Triggs (2005) HOG + improved multi-scale strategy
    """
    )

    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")

    # Detection parameters (Strong pass)
    st.sidebar.subheader("🎯 Detection (Strong Pass)")
    win_stride = st.sidebar.slider("Window Stride", 2, 16, 4, 2)
    padding = st.sidebar.slider("Padding", 0, 32, 8, 4)
    hit_threshold = st.sidebar.slider("Hit Threshold", 0.0, 2.0, 0.6, 0.05)
    min_final_score = st.sidebar.slider("Min Final Score", 0.0, 2.0, 0.6, 0.05)

    # Multi-scale parameters
    st.sidebar.subheader("📏 Multi-Scale Settings")
    min_person_px = st.sidebar.slider("Min Person Height (px)", 20, 120, 40, 5)
    max_person_px = st.sidebar.slider("Max Person Height (px)", 80, 500, 220, 10)
    num_scales = st.sidebar.slider("Number of Scales", 4, 15, 6, 1)

    # Post-processing
    st.sidebar.subheader("🔧 Post-Processing")
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.05, 0.6, 0.2, 0.01)
    min_box_area = st.sidebar.slider("Min Box Area", 200, 8000, 1500, 100)

    # Filtering
    st.sidebar.subheader("🖼️ Aspect Ratio Filtering")
    min_aspect_ratio = st.sidebar.slider("Min Aspect Ratio", 0.10, 0.80, 0.35, 0.01)
    max_aspect_ratio = st.sidebar.slider("Max Aspect Ratio", 0.20, 1.20, 0.65, 0.01)

    # Dual-pass
    st.sidebar.subheader("🧪 Dual-Pass Strategy")
    weak_scale = st.sidebar.slider(
        "Weak Pass Strength",
        0.2,
        1.0,
        0.5,
        0.05,
        help="Lower = more aggressive recall boost",
    )

    # Preprocessing
    st.sidebar.subheader("🖼️ Preprocessing")
    use_preprocessing = st.sidebar.checkbox("Enable Preprocessing", True)

    if use_preprocessing:
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.1, 0.05)
        contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.2, 0.05)
        sharpen = st.sidebar.checkbox("Sharpen", True)
    else:
        brightness = 1.0
        contrast = 1.0
        sharpen = False

    # Info expandables
    with st.sidebar.expander("💡 Parameter Guide"):
        st.markdown(
            """
        **Hit Threshold:** Higher = fewer, more confident detections
        **Min Final Score:** Final confidence cutoff
        **NMS Threshold:** Lower = better separation
        **Weak Pass Strength:** Lower = catch more difficult cases
        
        **Recommended for mall CCTV:**
        - Hit Threshold: 0.5-0.7
        - NMS: 0.15-0.25
        - Weak Strength: 0.4-0.6
        """
        )

    with st.sidebar.expander("📚 References"):
        st.markdown(
            """
        1. Dalal & Triggs (2005) - HOG Features
        2. Felzenszwalb et al. (2010) - DPM
        3. OpenCV HOGDescriptor
        4. Area-based density estimation
        """
        )

    # Initialize detector
    if "detector" not in st.session_state:
        st.session_state.detector = ImprovedHOGDetector()

    detector = st.session_state.detector

    # File upload
    uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("👆 Upload an image to start detection")
        return

    # Load image
    pil_img = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_img)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Create parameter objects
    params_strong = DetectionParams(
        use_preprocessing=use_preprocessing,
        brightness=brightness,
        contrast=contrast,
        sharpen=sharpen,
        win_stride=(win_stride, win_stride),
        padding=(padding, padding),
        hit_threshold=hit_threshold,
        min_final_score=min_final_score,
        min_person_px=min_person_px,
        max_person_px=max_person_px,
        num_scales=num_scales,
        min_box_area=min_box_area,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        nms_threshold=nms_threshold,
    )

    params_weak = create_weak_params(params_strong, weak_scale)

    # Display images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Original Image")
        st.image(pil_img, use_container_width=True)

    # Run detection
    with st.spinner("🔍 Detecting people (dual-pass)..."):
        boxes, weights = dual_pass_detect(
            detector, image_bgr, params_strong, params_weak
        )

        # Calculate density
        density_ratio, crowd_level = estimate_crowd_density(boxes, image_bgr.shape)
        people_count = len(boxes)

        # Visualize
        result_bgr = detector.visualize(
            image_bgr,
            boxes,
            weights,
            people_text=f"People: {people_count}",
            crowd_text=f"Crowd: {crowd_level}",
            density_text=f"Density: {density_ratio:.3f}",
        )
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("🎯 Detection Results")
        st.image(result_rgb, use_container_width=True)

    # Statistics
    st.markdown("---")
    st.subheader("📊 Detection Statistics")

    col3, col4, col5, col6 = st.columns(4)

    avg_confidence = float(np.mean(weights)) if weights else 0.0

    with col3:
        st.metric("People Detected", people_count)
    with col4:
        level_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Very High": "🔴"}
        st.metric("Crowd Level", f"{level_emoji.get(crowd_level, '')} {crowd_level}")
    with col5:
        st.metric("Area Density", f"{density_ratio:.3f}")
    with col6:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    # Detection details
    if boxes:
        with st.expander("📋 Detection Details"):
            for i, (box, weight) in enumerate(zip(boxes, weights)):
                x, y, w, h = [int(v) for v in box]
                aspect = w / (h + 1e-6)
                st.text(
                    f"Person {i+1:2d}: ({x:4d},{y:4d}) | "
                    f"Size: {w:3d}×{h:3d} | AR: {aspect:.2f} | "
                    f"Conf: {weight:.2f}"
                )

    # Download button
    st.markdown("---")
    buf = io.BytesIO()
    Image.fromarray(result_rgb).save(buf, format="PNG")
    st.download_button(
        "⬇️ Download Result Image", buf.getvalue(), "detection_result.png", "image/png"
    )


if __name__ == "__main__":
    main()
