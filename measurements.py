"""Posture measurement formulas. Owner: Roy.

MediaPipe Pose landmark indices reference:
    0  nose          11 L shoulder   23 L hip      27 L ankle
    7  L ear         12 R shoulder   24 R hip      28 R ankle
    8  R ear         13 L elbow      25 L knee
                     14 R elbow      26 R knee
"""

import math
import numpy as np


def _angle_between_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the angle p1-p2-p3 at p2. Returns degrees in [0, 180]."""
    # Vector p2 -> p1
    v1 = p1[:2] - p2[:2]
    # Vector p2 -> p3
    v2 = p3[:2] - p2[:2]
    
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return float(angle)


def measure_front(landmarks: np.ndarray) -> dict:
    """Compute front-view posture metrics from landmarks.

    Returns:
        {
          "shoulder_height_diff_pct": float,  # |y_L - y_R| / shoulder_width * 100
          "pelvic_tilt_deg": float,           # signed: + = left hip lower, - = right hip lower
          "knee_angle_deg": float,            # avg hip-knee-ankle inner angle (L, R)
          "knee_alignment": str,              # "normal" | "X-leg" | "O-leg"
        }
    """
    l_shoulder, r_shoulder = landmarks[11], landmarks[12]
    l_hip, r_hip = landmarks[23], landmarks[24]

    # Shoulder height diff pct
    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    if shoulder_width > 0:
        shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
        shoulder_height_diff_pct = (shoulder_diff / shoulder_width) * 100
    else:
        shoulder_height_diff_pct = 0.0

    # Signed pelvic tilt. Image y grows downward, so positive dy means left
    # hip drawn lower in the image — i.e. the person's left hip is lower.
    dy_hip = l_hip[1] - r_hip[1]
    dx_hip = abs(l_hip[0] - r_hip[0])
    pelvic_tilt_deg = math.degrees(math.atan2(dy_hip, dx_hip))

    l_knee_angle = _angle_between_points(landmarks[23], landmarks[25], landmarks[27])
    r_knee_angle = _angle_between_points(landmarks[24], landmarks[26], landmarks[28])
    knee_angle_deg = (l_knee_angle + r_knee_angle) / 2.0

    # X/O leg from knee-vs-ankle distance ratio (independent of knee bend)
    knee_dist = abs(landmarks[25][0] - landmarks[26][0])
    ankle_dist = abs(landmarks[27][0] - landmarks[28][0])
    if ankle_dist < 1e-6:
        knee_alignment = "normal"
    else:
        ratio = knee_dist / ankle_dist
        if ratio < 0.7:
            knee_alignment = "X-leg"
        elif ratio > 1.5:
            knee_alignment = "O-leg"
        else:
            knee_alignment = "normal"

    return {
        "shoulder_height_diff_pct": float(shoulder_height_diff_pct),
        "pelvic_tilt_deg": float(pelvic_tilt_deg),
        "knee_angle_deg": float(knee_angle_deg),
        "knee_alignment": knee_alignment,
    }


def measure_side(landmarks: np.ndarray) -> dict:
    """Compute side-view posture metrics from landmarks.

    Pick the side facing the camera by comparing visibility of landmarks 7 vs 8.

    Returns:
        {
          "head_forward_pct": float,           # (ear_x - shoulder_x) / shoulder-hip distance
          "body_line_deviation_deg": float,    # shoulder->ankle line angle from vertical
          "thigh_forward_tilt_deg": float,     # hip->knee line angle from vertical (proxy for pelvic tilt)
        }
    """
    # Pick side (index 3 is visibility)
    if landmarks[7, 3] > landmarks[8, 3]:  # Left side faces camera
        ear = landmarks[7]
        shoulder = landmarks[11]
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]
    else:                                  # Right side faces camera
        ear = landmarks[8]
        shoulder = landmarks[12]
        hip = landmarks[24]
        knee = landmarks[26]
        ankle = landmarks[28]

    # Note: X goes left-to-right on image. To calculate head forward, 
    # we need to know which way the person is facing.
    # Nose (0) x vs ear x determines facing direction.
    nose_x = landmarks[0, 0]
    facing_right = nose_x > ear[0]

    ear_x, shoulder_x = ear[0], shoulder[0]
    shoulder_hip_dist = np.linalg.norm(shoulder[:2] - hip[:2])
    
    if shoulder_hip_dist > 0:
        if facing_right:
            head_forward = ear_x - shoulder_x
        else:
            head_forward = shoulder_x - ear_x
        head_forward_pct = head_forward / shoulder_hip_dist
    else:
        head_forward_pct = 0.0

    # Body line deviation (angle of shoulder->ankle from vertical)
    # Vertical vector: (0, 1) (downwards on image)
    dx_body = ankle[0] - shoulder[0]
    dy_body = ankle[1] - shoulder[1]
    # Angle from vertical (Y-axis) -> atan2(|dx|, dy)
    body_line_deviation_deg = abs(math.degrees(math.atan2(abs(dx_body), abs(dy_body))))

    # Thigh tilt from vertical (proxy for pelvic anterior/posterior tilt)
    dx_thigh = knee[0] - hip[0]
    dy_thigh = knee[1] - hip[1]
    thigh_forward_tilt_deg = math.degrees(math.atan2(abs(dx_thigh), abs(dy_thigh)))

    return {
        "head_forward_pct": float(head_forward_pct),
        "body_line_deviation_deg": float(body_line_deviation_deg),
        "thigh_forward_tilt_deg": float(thigh_forward_tilt_deg),
    }
