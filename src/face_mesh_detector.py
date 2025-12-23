"""
Face Mesh Detector Module
Wraps MediaPipe Face Mesh functionality for detecting facial landmarks.
"""

import cv2
import mediapipe as mp
from typing import Optional


class FaceMeshDetector:
    """Wrapper class for MediaPipe FaceMesh detection."""
    
    def __init__(self, max_faces: int = 1):
        """
        Initialize MediaPipe FaceMesh model.
        
        :param max_faces: Maximum number of faces to detect (default: 1).
        """
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Create Face Mesh object
        # static_image_mode=False for video processing
        # max_num_faces can be adjusted based on requirements
        # min_detection_confidence=0.5 for reliable detection
        # min_tracking_confidence=0.5 for stable tracking
        # refine_landmarks=True to get iris landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect(self, frame_bgr):
        """
        Process a BGR frame and detect face landmarks.
        
        :param frame_bgr: Input frame in BGR format (OpenCV default).
        :return: face_landmarks of the first detected face, or None if no face detected.
        """
        # Convert the BGR image to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect face mesh
        results = self.face_mesh.process(rgb_frame)
        
        # Return the first face landmarks if detected, otherwise None
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None
    
    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()




