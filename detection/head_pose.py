import numpy as np
import math
import cv2

# These are key 3D facial landmarks indices from MediaPipe FaceMesh
# (nose tip, forehead, chin, left/right eye corners)
LANDMARK_NOSE = 1
LANDMARK_CHIN = 152
LANDMARK_LEFT_EYE = 33
LANDMARK_RIGHT_EYE = 263
LANDMARK_MOUTH_LEFT = 61
LANDMARK_MOUTH_RIGHT = 291


def get_head_pose(landmarks, frame_w, frame_h):
    """
    Estimate head pose angles (pitch, yaw, roll) in degrees.
    Input:
        landmarks: MediaPipe face landmarks (468 points)
        frame_w, frame_h: image dimensions
    Output:
        pitch, yaw, roll
    """

    # Convert MediaPipe normalized coordinates to pixel coordinates
    def to_pixel(point):
        return np.array([point.x * frame_w, point.y * frame_h])

    image_points = np.array([
        to_pixel(landmarks[LANDMARK_NOSE]),        # Nose tip
        to_pixel(landmarks[LANDMARK_CHIN]),        # Chin
        to_pixel(landmarks[LANDMARK_LEFT_EYE]),    # Left eye corner
        to_pixel(landmarks[LANDMARK_RIGHT_EYE]),   # Right eye corner
        to_pixel(landmarks[LANDMARK_MOUTH_LEFT]),  # Left mouth
        to_pixel(landmarks[LANDMARK_MOUTH_RIGHT])  # Right mouth
    ], dtype="double")

    # 3D model points of human face - approximate values
    model_points = np.array([
        [0.0, 0.0, 0.0],           # Nose
        [0.0, -63.6, -12.5],       # Chin
        [-43.3, 32.7, -26.0],      # Left eye
        [43.3, 32.7, -26.0],       # Right eye
        [-28.9, -28.9, -24.1],     # Left mouth
        [28.9, -28.9, -24.1]       # Right mouth
    ])

    # Camera parameters
    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # No distortion

    # SolvePnP returns rotation + translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Convert rotation vector into rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calculate Euler angles
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    pitch = math.atan2(-rotation_matrix[2, 0], sy) * (180.0 / math.pi)
    yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * (180.0 / math.pi)
    roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * (180.0 / math.pi)

    return pitch, yaw, roll


def is_head_nodding(pitch, threshold_down=20):
    """
    Detect head-nodding (drowsiness).
    If pitch increases significantly (head tilts down), it means drowsiness.

    pitch > +20 degrees → head moving down → sleepy nod
    """
    return pitch > threshold_down
