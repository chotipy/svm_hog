import io
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from model_registry import ModelKey, MODEL_CONFIGS
from detector_factory import build_detector
from utils_detection import nms_xywh


def estimate_crowd_density(
    boxes: List[List[float]], image_shape: Tuple[int, int, int]
) -> tuple[float, str]:
    h, w = image_shape[:2]
    image_area = float(w * h)

    total_person_area = sum(float(bw * bh) for _, _, bw, bh in boxes)
    density_ratio = total_person_area / (image_area + 1e-9)

    if density_ratio < 0.03:
        level = "Low"
    elif density_ratio < 0.06:
        level = "Medium"
    elif density_ratio < 0.10:
        level = "High"
    else:
        level = "Very High"

    return density_ratio, level


def apply_theme(theme: str):
    if theme == "Light":
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
    else:
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

        h1, h2, h3 {
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
        </style>
        """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Crowd Detector", layout="wide")
    st.title("Advanced Crowd Detector")
    st.markdown(
        "<p style='color:#888; margin-top:-10px;'>Model-based detection with crowd metrics</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.header("⚙️ Configuration")

    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
    apply_theme(theme)

    model_choice = st.sidebar.selectbox(
        "Detection Model",
        options=[m.value for m in ModelKey],
        index=0,
    )
    model_key = ModelKey(model_choice)
    model_cfg = MODEL_CONFIGS[model_key]

    try:
        detector = build_detector(model_cfg)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    st.sidebar.success(f"Loaded: {model_key.value}")

    # ======================
    # Upload image
    # ======================
    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if uploaded is None:
        st.info("Upload image to run detection")
        st.stop()

    img_rgb = np.array(Image.open(uploaded).convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    col1.subheader("Original")
    col1.image(img_rgb, use_container_width=True)

    with st.spinner("Running detection..."):
        boxes: List[List[float]] = []
        scores: List[float] = []

        if model_key == ModelKey.OPENCV and hasattr(detector, "hog"):
            b, w = detector.hog.detectMultiScale(
                img_bgr,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            if len(b) > 0:
                boxes = b.astype(float).tolist()
                scores = w.flatten().astype(float).tolist()

        else:
            boxes, scores = detector.detect(img_bgr)

        # NMS (basic, same for all)
        boxes, scores = nms_xywh(boxes, scores, iou_thr=0.3)

        # Draw
        vis = img_bgr.copy()
        for (x, y, w, h), sc in zip(boxes, scores):
            x, y, w, h = map(int, [x, y, w, h])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{sc:.2f}",
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        col2.subheader("Detections")
        col2.image(vis_rgb, use_container_width=True)

        people_count = len(boxes)
        avg_conf = float(np.mean(scores)) if scores else 0.0
        density_ratio, crowd_level = estimate_crowd_density(boxes, img_bgr.shape)

        st.markdown("---")
        st.subheader("📊 Detection Metrics")

        c1, c2, c3, c4 = st.columns(4)

        emoji = {
            "Low": "🟢",
            "Medium": "🟡",
            "High": "🟠",
            "Very High": "🔴",
        }

        c1.metric("People Detected", people_count)
        c2.metric("Crowd Level", f"{emoji[crowd_level]} {crowd_level}")
        c3.metric("Area Density", f"{density_ratio:.3f}")
        c4.metric("Avg Confidence", f"{avg_conf:.2f}")

        buf = io.BytesIO()
        Image.fromarray(vis_rgb).save(buf, format="PNG")
        st.download_button(
            "⬇️ Download result",
            buf.getvalue(),
            "result.png",
            "image/png",
        )


if __name__ == "__main__":
    main()
