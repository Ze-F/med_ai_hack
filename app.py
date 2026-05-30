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

st.set_page_config(page_title="Posture Analyzer", layout="centered", page_icon="🧘")

st.markdown(
    """
    <style>
    /* hide streamlit chrome for a clean demo look */
    #MainMenu, footer, header, .stDeployButton { visibility: hidden; height: 0; }

    /* warmer page padding + max width */
    .block-container { padding-top: 2rem !important; padding-bottom: 4rem !important;
                       max-width: 980px !important; }

    /* hero */
    .hero-title { font-size: 44px; font-weight: 700; margin: 0;
                  color: #2F3D33; letter-spacing: -1px; }
    .hero-sub   { font-size: 17px; color: #6B7B70; margin-top: 6px; margin-bottom: 32px; }

    /* wizard stepper */
    .stepper { display: flex; align-items: center; justify-content: center;
               margin: 8px auto 36px auto; max-width: 620px; }
    .step    { display: flex; flex-direction: column; align-items: center;
               min-width: 130px; }
    .step .circle { width: 36px; height: 36px; border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-weight: 600; font-size: 15px; transition: all .2s;
                    border: 2px solid transparent; }
    .step .label  { font-size: 13px; margin-top: 8px; color: #8A9690; font-weight: 500; }

    .step.todo   .circle { background: #EFEAE2; color: #B5AB9E; }
    .step.active .circle { background: #9CAF88; color: white;
                           box-shadow: 0 0 0 6px rgba(156,175,136,.18); }
    .step.active .label  { color: #2F3D33; font-weight: 600; }
    .step.done   .circle { background: #6F8B61; color: white; }
    .step.done   .label  { color: #5A6B5D; }

    .line { flex: 1; height: 2px; background: #E8E2D6; max-width: 80px;
            margin: 0 -10px; margin-bottom: 22px; }
    .line.done { background: #9CAF88; }

    /* upload box softening */
    [data-testid="stFileUploader"] section {
        border: 2px dashed #D4CBB9 !important;
        background: #FBF8F1 !important;
        border-radius: 14px !important;
        padding: 28px !important;
    }
    [data-testid="stFileUploader"] section:hover { border-color: #9CAF88 !important; }

    /* metric cards softer */
    [data-testid="stMetric"] { background: white; border-radius: 14px;
                               padding: 14px 16px;
                               box-shadow: 0 1px 3px rgba(60,60,60,.04),
                                           0 0 0 1px #EFEAE2; }
    [data-testid="stMetricLabel"] { color: #6B7B70 !important; font-size: 13px !important; }

    /* primary button warm */
    .stButton > button[kind="primary"] {
        background: #9CAF88; border: none; padding: 12px 28px;
        border-radius: 12px; font-weight: 600; font-size: 16px;
        box-shadow: 0 2px 8px rgba(156,175,136,.25);
    }
    .stButton > button[kind="primary"]:hover { background: #88A073; }

    .legend { font-size:12px; color:#8A9690; margin-top: 6px; }
    .legend-dot { display:inline-block; width:8px; height:8px;
                  border-radius:999px; margin:0 4px 0 10px; vertical-align:middle; }

    .section-title { font-size: 20px; font-weight: 600; color: #2F3D33;
                     margin: 28px 0 4px 0; }
    .section-sub   { font-size: 14px; color: #8A9690; margin-bottom: 16px; }

    hr { border-color: #EFEAE2 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center;">
      <div class="hero-title">🧘 Posture Analyzer</div>
      <div class="hero-sub">Snap two photos. We'll spot what your body is asking you to change.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


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
    labels = ["Upload", "Verify landmarks", "Report"]
    parts = ['<div class="stepper">']
    for i, label in enumerate(labels, 1):
        state = "active" if i == current else ("done" if i < current else "todo")
        mark = "✓" if state == "done" else str(i)
        parts.append(
            f'<div class="step {state}">'
            f'  <div class="circle">{mark}</div>'
            f'  <div class="label">{label}</div>'
            f"</div>"
        )
        if i < len(labels):
            line_state = "done" if i < current else ""
            parts.append(f'<div class="line {line_state}"></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


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


if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

stepper_slot = st.empty()

# ====== STEP 1: UPLOAD ======
ready_for_editor = False
front_image = side_image = None
front_landmarks = side_landmarks = None
front_annotated = side_annotated = None

st.markdown('<div class="section-title">📷 Upload your photos</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Two clear, full-body shots: facing the camera, then turned 90° to one side.</div>',
    unsafe_allow_html=True,
)

col_front, col_side = st.columns(2)
with col_front:
    front_file = st.file_uploader("Front view", type=["jpg", "jpeg", "png"], key="front")
with col_side:
    side_file = st.file_uploader("Side view", type=["jpg", "jpeg", "png"], key="side")

if front_file and side_file:
    front_image, front_landmarks, front_annotated = _run_detection(front_file.getvalue())
    side_image, side_landmarks, side_annotated = _run_detection(side_file.getvalue())
    if front_landmarks is None:
        st.error("❌ No person detected in front photo. Try a clearer full-body shot.")
    if side_landmarks is None:
        st.error("❌ No person detected in side photo. Try a clearer full-body shot.")
    ready_for_editor = front_landmarks is not None and side_landmarks is not None

# ====== STEP 2: VERIFY LANDMARKS ======
adjusted_front = adjusted_side = None
if ready_for_editor:
    st.markdown(
        '<div class="section-title">✏️ Verify the landmarks</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Pose detection isn\'t perfect. Drag any dot to nudge it onto the right joint.</div>',
        unsafe_allow_html=True,
    )
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

    st.write("")
    if st.button("Analyze posture", type="primary", use_container_width=True):
        st.session_state.report_generated = True
        front_metrics = measure_front(adjusted_front)
        side_metrics = measure_side(adjusted_side)

        st.markdown(
            '<div class="section-title">📊 Your measurements</div>',
            unsafe_allow_html=True,
        )
        _show_metrics(front_metrics, side_metrics)

        st.markdown(
            '<div class="section-title">📋 Posture report</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Generating report with Claude..."):
            report = generate_report(
                front_annotated, side_annotated, front_metrics, side_metrics
            )
        st.markdown(report)

# Render stepper into the top slot now that we know the state
current_step = 3 if st.session_state.report_generated else (2 if ready_for_editor else 1)
with stepper_slot:
    _step_indicator(current_step)
