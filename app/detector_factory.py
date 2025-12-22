import streamlit as st
from model_registry import ModelConfig
from backends.opencv_hog import OpenCVHOGDetector
from backends.svm_window import SVMWindowDetector


@st.cache_resource(show_spinner=False)
def build_detector(model_config: ModelConfig):
    if model_config.backend == "opencv":
        return OpenCVHOGDetector()

    if model_config.backend == "svm":
        if not model_config.model_dir:
            raise ValueError("model_dir is required for svm backend")

        return SVMWindowDetector(
            model_dir=model_config.model_dir,
            clf_filename=model_config.clf_filename or "",
            scaler_or_bundle_filename=model_config.scaler_or_bundle_filename,
            feature_type=model_config.feature_type,
            config_filename=model_config.config_filename,
            default_window_size=model_config.window_size,
            default_step_size=model_config.step_size,
            default_min_conf=model_config.min_confidence,
            default_nms=model_config.nms_threshold,
        )

    raise ValueError(f"Unknown backend: {model_config.backend}")
