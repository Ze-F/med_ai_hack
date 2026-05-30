"""Posture measurement formulas. Owner: Roy.

MediaPipe Pose landmark indices reference:
    0  nose          11 L shoulder   23 L hip      27 L ankle
    7  L ear         12 R shoulder   24 R hip      28 R ankle
    8  R ear         13 L elbow      25 L knee
                     14 R elbow      26 R knee
"""

import numpy as np


def measure_front(landmarks: np.ndarray) -> dict:
    """Compute front-view posture metrics from landmarks.

    Returns:
        {
          "shoulder_height_diff_pct": float,  # |y_L - y_R| / shoulder_width * 100
          "pelvic_tilt_deg": float,           # angle of L-R hip line from horizontal
          "knee_angle_deg": float,            # avg hip-knee-ankle inner angle (L, R)
          "knee_alignment": str,              # "normal" | "X-leg" | "O-leg"
        }
    """
    # TODO Roy: implement using landmark indices above
    return {
        "shoulder_height_diff_pct": 0.0,
        "pelvic_tilt_deg": 0.0,
        "knee_angle_deg": 180.0,
        "knee_alignment": "normal",
    }


def measure_side(landmarks: np.ndarray) -> dict:
    """Compute side-view posture metrics from landmarks.

    Pick the side facing the camera by comparing visibility of landmarks 7 vs 8.

    Returns:
        {
          "head_forward_pct": float,           # (ear_x - shoulder_x) / shoulder-hip distance
          "body_line_deviation_deg": float,    # shoulder->ankle line angle from vertical
          "pelvic_rotation_deg": float,        # hip->knee line angle from vertical
        }
    """
    # TODO Roy: implement
    return {
        "head_forward_pct": 0.0,
        "body_line_deviation_deg": 0.0,
        "pelvic_rotation_deg": 0.0,
    }
