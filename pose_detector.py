"""MediaPipe Pose wrapper. Owner: Roy."""

import numpy as np
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
    # TODO Roy: replace mock with MediaPipe Pose
    #   1. mp_pose = mediapipe.solutions.pose.Pose(static_image_mode=True)
    #   2. results = mp_pose.process(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    #   3. landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility]
    #                            for lm in results.pose_landmarks.landmark])
    #   4. draw with mp.solutions.drawing_utils.draw_landmarks(...)
    #   5. convert back to PIL.Image

    mock_landmarks = np.tile([0.5, 0.5, 0.0, 1.0], (33, 1))
    return mock_landmarks, image
