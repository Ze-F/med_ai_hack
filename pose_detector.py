"""MediaPipe Pose wrapper. Owner: Roy."""

import numpy as np
import mediapipe as mp
from PIL import Image


def detect_pose(image: Image.Image) -> tuple[np.ndarray | None, Image.Image]:
    """Detect body landmarks in a photo.

    Args:
        image: PIL image of a person standing.

    Returns:
        (landmarks, annotated_image)
        landmarks: shape (33, 4) — [x, y, z, visibility], normalized 0-1.
                   None if no person detected.
        annotated_image: original image with skeleton drawn on top.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize MediaPipe Pose for static images
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:
        
        # MediaPipe expects RGB images. Convert PIL image to numpy array (RGB)
        image_np = np.array(image.convert("RGB"))
        
        # Process the image to find the pose
        results = pose.process(image_np)

        # If no pose is detected, return None and the original image
        if not results.pose_landmarks:
            return None, image

        # Extract landmarks into a (33, 4) numpy array
        landmarks = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ])

        # Draw the pose annotation on a copy of the image
        annotated_image_np = image_np.copy()
        mp_drawing.draw_landmarks(
            annotated_image_np,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Convert the annotated numpy array back to a PIL Image
        annotated_image = Image.fromarray(annotated_image_np)

        return landmarks, annotated_image
