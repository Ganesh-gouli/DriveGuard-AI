# detection/mouth_detection.py
import numpy as np

# mouth indices used from MediaPipe
# upper lip: 13, lower lip: 14 ; use pair indices to compute MAR
MOUTH_INDICES = [13, 14, 78, 308, 82, 312]  # approximate useful points

def mar_from_landmarks(landmarks):
    """
    Simple surrogate MAR using pairs - not a canonical formula but works for demo.
    landmarks: list of (x,y) positions
    returns a float (larger -> mouth more open)
    """
    try:
        up = np.array(landmarks[MOUTH_INDICES[0]])
        down = np.array(landmarks[MOUTH_INDICES[1]])
        left = np.array(landmarks[MOUTH_INDICES[2]])
        right = np.array(landmarks[MOUTH_INDICES[3]])
        vertical = np.linalg.norm(up - down)
        horizontal = np.linalg.norm(left - right)
        if horizontal == 0:
            return 0.0
        mar = vertical / horizontal
        return mar
    except Exception as e:
        return 0.0
