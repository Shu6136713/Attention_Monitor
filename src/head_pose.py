"""
Head Pose Estimation Module
Estimates head pose (yaw, pitch, roll) from facial landmarks using PnP algorithm.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class HeadPoseEstimator:
    """Estimates head pose angles from 2D facial landmarks."""
    
    def __init__(self):
        """
        Initialize the head pose estimator with a 3D face model.
        
        The 3D model points represent key facial landmarks in a normalized
        coordinate system (roughly centered at the nose).
        """
        # 3D model points of a generic face (in centimeters)
        # These correspond to specific MediaPipe face mesh landmark indices
        # Order: Nose tip, Chin, Left eye left corner, Right eye right corner,
        #        Left mouth corner, Right mouth corner
        # Adjusted to match OpenCV/MediaPipe Coordinate System (Y is Down)
        # Nose is (0,0,0)
        # Chin is below nose -> Positive Y
        # Eyes are above nose -> Negative Y
        # Mouth is below nose -> Positive Y
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (index 1)
            (0.0, 6.3, -1.2),            # Chin (index 152) - was -6.3
            (-4.3, -2.5, -1.7),          # Left eye left corner (index 33) - was 2.5
            (4.3, -2.5, -1.7),           # Right eye right corner (index 263) - was 2.5
            (-2.9, 2.5, -1.4),           # Left mouth corner (index 61) - was -2.5
            (2.9, 2.5, -1.4)             # Right mouth corner (index 291) - was -2.5
        ], dtype=np.float64)
        
        # Corresponding MediaPipe Face Mesh landmark indices
        # MediaPipe Face Mesh has 468 landmarks (0-467)
        self.landmark_indices = [1, 152, 33, 263, 61, 291]
    
    def estimate(self, landmarks, frame_shape: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
        """
        Compute head pose from 2D facial landmarks.
        
        :param landmarks: MediaPipe Face Mesh landmarks (468 points).
        :param frame_shape: Tuple of (height, width) of the frame.
        :return: Tuple of (yaw, pitch, roll) in degrees, or None if landmarks are invalid.
        """
        # Return None if no landmarks provided
        if landmarks is None:
            return None
        
        # Extract frame dimensions
        height, width = frame_shape
        
        # Extract 2D image points from MediaPipe landmarks
        image_points_2d = []
        for idx in self.landmark_indices:
            # MediaPipe landmarks are normalized (0.0 to 1.0)
            # Convert to pixel coordinates
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            image_points_2d.append([x, y])
        
        image_points_2d = np.array(image_points_2d, dtype=np.float64)
        
        # Camera internals (focal length and optical center)
        # Assuming a generic webcam with focal length approximately equal to frame width
        focal_length = width
        center = (width / 2, height / 2)
        
        # Camera matrix
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3d,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # If PnP fails, return None
        if not success:
            return None
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles from rotation matrix
        yaw, pitch, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        # Convert from radians to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)
        
        return yaw, pitch, roll
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a rotation matrix to Euler angles (yaw, pitch, roll).
        
        Uses the convention: yaw (Y-axis), pitch (X-axis), roll (Z-axis).
        
        :param R: 3x3 rotation matrix.
        :return: Tuple of (yaw, pitch, roll) in radians.
        """
        # Check for gimbal lock
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            # Normal case
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            # Gimbal lock case
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = 0
        
        return yaw, pitch, roll
