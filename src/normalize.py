import numpy as np


class LandmarkNormalizer:
    """
    Normalize MediaPipe hand landmarks.

    Output:
        63 values (21 landmarks × 3 coordinates)

    Steps:
        1. Move wrist to the origin.
        2. Scale hand size to make it independent of distance.
    """

    def normalize(self, landmarks):

        # Convert to NumPy array (21 x 3)
        points = np.array(landmarks, dtype=np.float32).reshape(21, 3)

        # -------------------------------------------------
        # Step 1: Move wrist to (0,0,0)
        # -------------------------------------------------
        wrist = points[0]
        points = points - wrist

        # -------------------------------------------------
        # Step 2: Normalize by maximum distance
        # -------------------------------------------------
        distances = np.linalg.norm(points, axis=1)
        max_distance = np.max(distances)

        if max_distance > 0:
            points = points / max_distance

        # -------------------------------------------------
        # Return as 63 values
        # -------------------------------------------------
        return points.flatten()
