"""Streamlit posture analyzer UI. Owner: Ze."""

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from llm_report import generate_report
from measurements import measure_front, measure_side
from pose_detector import detect_pose

load_dotenv()

st.set_page_config(page_title="Posture Analyzer", layout="wide")
st.title("🧍 Posture Analyzer")
st.caption("Upload a front-view and a side-view standing photo for posture analysis.")


@st.cache_data(show_spinner=False)
def _run_detection(image_bytes: bytes):
    image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    landmarks, annotated = detect_pose(image)
    return landmarks, annotated


col_front, col_side = st.columns(2)

with col_front:
    st.subheader("Front view")
    front_file = st.file_uploader("Upload front photo", type=["jpg", "jpeg", "png"], key="front")
    front_landmarks = front_annotated = None
    if front_file:
        front_landmarks, front_annotated = _run_detection(front_file.getvalue())
        st.image(front_annotated, use_column_width=True)
        if front_landmarks is None:
            st.error("No person detected in front photo.")

with col_side:
    st.subheader("Side view")
    side_file = st.file_uploader("Upload side photo", type=["jpg", "jpeg", "png"], key="side")
    side_landmarks = side_annotated = None
    if side_file:
        side_landmarks, side_annotated = _run_detection(side_file.getvalue())
        st.image(side_annotated, use_column_width=True)
        if side_landmarks is None:
            st.error("No person detected in side photo.")

st.divider()

ready = front_landmarks is not None and side_landmarks is not None

if st.button("Analyze posture", disabled=not ready, type="primary"):
    front_metrics = measure_front(front_landmarks)
    side_metrics = measure_side(side_landmarks)

    st.subheader("Measurements")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("**Front view**")
        st.json(front_metrics)
    with m_col2:
        st.markdown("**Side view**")
        st.json(side_metrics)

    st.subheader("Report")
    with st.spinner("Generating report..."):
        report = generate_report(front_annotated, side_annotated, front_metrics, side_metrics)
    st.markdown(report)
