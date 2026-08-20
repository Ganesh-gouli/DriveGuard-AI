# detection/eye_detection.py
import numpy as np

# using MediaPipe FaceMesh indices (commonly used)
# left eye: [33,160,158,133,153,144]
# right eye: [362,385,387,263,373,380]
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def ear_from_landmarks(landmarks, eye_indices):
    """
    landmarks: list of (x,y) tuples normalized or pixel coords
    eye_indices: 6 indices
    """
    p = [landmarks[i] for i in eye_indices]
    # vertical distances
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    # horizontal distance
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def avg_ear(landmarks):
    left = ear_from_landmarks(landmarks, LEFT_EYE)
    right = ear_from_landmarks(landmarks, RIGHT_EYE)
    return (left + right) / 2.0
