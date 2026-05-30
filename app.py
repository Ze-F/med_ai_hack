"""Streamlit posture analyzer UI. Owner: Ze."""

import io

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from llm_report import generate_report
from measurements import measure_front, measure_side
from pose_detector import detect_pose

load_dotenv()

st.set_page_config(page_title="Posture Analyzer", layout="wide", page_icon="🧍")

st.markdown(
    """
    <style>
    .step-pill { display:inline-block; padding:4px 12px; margin-right:8px;
                 border-radius:999px; font-size:13px; font-weight:600; }
    .step-active { background:#2563eb; color:white; }
    .step-done   { background:#16a34a; color:white; }
    .step-todo   { background:#e5e7eb; color:#6b7280; }
    .legend { font-size:13px; color:#6b7280; }
    .legend-dot { display:inline-block; width:10px; height:10px;
                  border-radius:999px; margin:0 4px 0 12px; vertical-align:middle; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧍 Posture Analyzer")
st.caption("Upload front + side standing photos. Adjust landmarks if needed. Get an AI posture report.")


FRONT_LANDMARKS = [
    (11, "L shoulder", "#3b82f6"),
    (12, "R shoulder", "#3b82f6"),
    (23, "L hip", "#ef4444"),
    (24, "R hip", "#ef4444"),
    (25, "L knee", "#22c55e"),
    (26, "R knee", "#22c55e"),
    (27, "L ankle", "#f97316"),
    (28, "R ankle", "#f97316"),
]

THRESHOLDS = {
    "shoulder_height_diff_pct": (2.0, 5.0, "%"),
    "pelvic_tilt_deg": (3.0, 6.0, "°"),
    "head_forward_pct": (0.10, 0.15, ""),
    "body_line_deviation_deg": (5.0, 10.0, "°"),
    "thigh_forward_tilt_deg": (5.0, 10.0, "°"),
}

METRIC_LABELS = {
    "shoulder_height_diff_pct": "Shoulder asymmetry",
    "pelvic_tilt_deg": "Pelvic tilt",
    "knee_angle_deg": "Knee angle",
    "knee_alignment": "Knee alignment",
    "head_forward_pct": "Head forward",
    "body_line_deviation_deg": "Body line",
    "thigh_forward_tilt_deg": "Thigh tilt",
}


def _step_indicator(current: int):
    labels = ["1. Upload", "2. Verify landmarks", "3. Report"]
    html = ""
    for i, label in enumerate(labels, 1):
        cls = "step-active" if i == current else ("step-done" if i < current else "step-todo")
        html += f'<span class="step-pill {cls}">{label}</span>'
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(show_spinner="Detecting pose...")
def _run_detection(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    landmarks, annotated = detect_pose(image)
    return image, landmarks, annotated


def _side_landmark_set(landmarks: np.ndarray):
    if landmarks[7, 3] >= landmarks[8, 3]:
        return [
            (7, "Ear (L)", "#a855f7"),
            (11, "Shoulder", "#3b82f6"),
            (23, "Hip", "#ef4444"),
            (25, "Knee", "#22c55e"),
            (27, "Ankle", "#f97316"),
        ]
    return [
        (8, "Ear (R)", "#a855f7"),
        (12, "Shoulder", "#3b82f6"),
        (24, "Hip", "#ef4444"),
        (26, "Knee", "#22c55e"),
        (28, "Ankle", "#f97316"),
    ]


def _landmark_editor(
    image: Image.Image,
    landmarks: np.ndarray,
    landmark_set: list[tuple[int, str, str]],
    key: str,
    max_width: int = 480,
) -> np.ndarray:
    img_w, img_h = image.size
    scale = min(1.0, max_width / img_w)
    canvas_w = int(img_w * scale)
    canvas_h = int(img_h * scale)
    bg = image.copy()
    bg.thumbnail((canvas_w, canvas_h))

    radius = 8
    objects = []
    for idx, label, color in landmark_set:
        cx = float(landmarks[idx, 0]) * canvas_w
        cy = float(landmarks[idx, 1]) * canvas_h
        objects.append({
            "type": "circle",
            "left": cx - radius,
            "top": cy - radius,
            "radius": radius,
            "fill": color,
            "stroke": "#ffffff",
            "strokeWidth": 2,
            "originX": "left",
            "originY": "top",
            "scaleX": 1,
            "scaleY": 1,
        })

    result = st_canvas(
        background_image=bg,
        initial_drawing={"version": "4.4.0", "objects": objects},
        drawing_mode="transform",
        width=canvas_w,
        height=canvas_h,
        update_streamlit=True,
        stroke_width=0,
        key=key,
    )

    legend = "<span class='legend'>Drag any dot to correct it." + "".join(
        f"<span class='legend-dot' style='background:{c}'></span>{label}"
        for _, label, c in landmark_set
    ) + "</span>"
    st.markdown(legend, unsafe_allow_html=True)

    adjusted = landmarks.copy()
    if result.json_data and result.json_data.get("objects"):
        for (idx, _, _), obj in zip(landmark_set, result.json_data["objects"]):
            sx = obj.get("scaleX", 1) or 1
            sy = obj.get("scaleY", 1) or 1
            r_eff = obj.get("radius", radius) * max(sx, sy)
            cx = obj["left"] + r_eff
            cy = obj["top"] + r_eff
            adjusted[idx, 0] = cx / canvas_w
            adjusted[idx, 1] = cy / canvas_h
    return adjusted


def _severity(value, key: str) -> tuple[str, str]:
    """Return (emoji, label) based on metric value vs THRESHOLDS."""
    if key not in THRESHOLDS:
        return "ℹ️", "info"
    notable, significant, _ = THRESHOLDS[key]
    v = abs(value)
    if v >= significant:
        return "🔴", "significant"
    if v >= notable:
        return "⚠️", "notable"
    return "✅", "normal"


def _metric_card(label: str, value, key: str, unit: str = ""):
    if isinstance(value, (int, float)):
        emoji, sev = _severity(value, key)
        display_val = f"{value:.2f}{unit}" if isinstance(value, float) else f"{value}{unit}"
        st.metric(label=f"{emoji} {label}", value=display_val, help=f"Status: {sev}")
    else:
        st.metric(label=f"ℹ️ {label}", value=str(value))


def _show_metrics(front_metrics: dict, side_metrics: dict):
    st.subheader("Measurements")
    st.markdown("**Front view**")
    cols = st.columns(4)
    with cols[0]:
        _metric_card(METRIC_LABELS["shoulder_height_diff_pct"],
                     front_metrics["shoulder_height_diff_pct"], "shoulder_height_diff_pct", "%")
    with cols[1]:
        _metric_card(METRIC_LABELS["pelvic_tilt_deg"],
                     front_metrics["pelvic_tilt_deg"], "pelvic_tilt_deg", "°")
    with cols[2]:
        _metric_card(METRIC_LABELS["knee_angle_deg"],
                     front_metrics["knee_angle_deg"], "knee_angle_deg", "°")
    with cols[3]:
        _metric_card(METRIC_LABELS["knee_alignment"],
                     front_metrics["knee_alignment"], "knee_alignment")

    st.markdown("**Side view**")
    cols = st.columns(3)
    with cols[0]:
        _metric_card(METRIC_LABELS["head_forward_pct"],
                     side_metrics["head_forward_pct"], "head_forward_pct")
    with cols[1]:
        _metric_card(METRIC_LABELS["body_line_deviation_deg"],
                     side_metrics["body_line_deviation_deg"], "body_line_deviation_deg", "°")
    with cols[2]:
        _metric_card(METRIC_LABELS["thigh_forward_tilt_deg"],
                     side_metrics["thigh_forward_tilt_deg"], "thigh_forward_tilt_deg", "°")


# ====== STEP 1: UPLOAD ======
current_step = 1
ready_for_editor = False

col_front, col_side = st.columns(2)
with col_front:
    st.subheader("📷 Front view")
    front_file = st.file_uploader("Upload front photo", type=["jpg", "jpeg", "png"], key="front")
with col_side:
    st.subheader("📷 Side view")
    side_file = st.file_uploader("Upload side photo", type=["jpg", "jpeg", "png"], key="side")

if front_file and side_file:
    front_image, front_landmarks, front_annotated = _run_detection(front_file.getvalue())
    side_image, side_landmarks, side_annotated = _run_detection(side_file.getvalue())
    if front_landmarks is None:
        st.error("❌ No person detected in front photo. Try a clearer full-body shot.")
    if side_landmarks is None:
        st.error("❌ No person detected in side photo. Try a clearer full-body shot.")
    ready_for_editor = front_landmarks is not None and side_landmarks is not None
    if ready_for_editor:
        current_step = 2

st.divider()
_step_indicator(current_step)
st.divider()

# ====== STEP 2: VERIFY LANDMARKS ======
adjusted_front = adjusted_side = None
if ready_for_editor:
    st.subheader("✏️ Verify & adjust landmarks")
    st.caption("Pose detection may have small errors. Drag any dot to its correct position.")
    edit_col1, edit_col2 = st.columns(2)
    with edit_col1:
        st.markdown("**Front**")
        adjusted_front = _landmark_editor(
            front_image, front_landmarks, FRONT_LANDMARKS, key="front_editor"
        )
    with edit_col2:
        st.markdown("**Side**")
        side_set = _side_landmark_set(side_landmarks)
        adjusted_side = _landmark_editor(
            side_image, side_landmarks, side_set, key="side_editor"
        )

    st.divider()
    if st.button("🚀 Analyze posture", type="primary", use_container_width=True):
        current_step = 3
        front_metrics = measure_front(adjusted_front)
        side_metrics = measure_side(adjusted_side)

        _show_metrics(front_metrics, side_metrics)

        st.subheader("📋 Posture Report")
        with st.container(border=True):
            with st.spinner("Generating report with Claude..."):
                report = generate_report(
                    front_annotated, side_annotated, front_metrics, side_metrics
                )
            st.markdown(report)
