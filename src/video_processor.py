"""
Video Processor Module
High-level frame processing that orchestrates detection and visualization.
"""

from src.face_mesh_detector import FaceMeshDetector
from src.gaze_logic import OrientationEstimator, AttentionLogic, GazeState
from src.utils.visualizer import draw_face_mesh, draw_attention_overlay


class VideoProcessor:
    """High-level processor that coordinates face detection and drawing."""
    
    def __init__(self):
        """Initialize the video processor with FaceMesh detector, orientation estimator, and attention logic."""
        self.face_detector = FaceMeshDetector()
        self.orientation_estimator = OrientationEstimator()
        self.attention_logic = AttentionLogic(
            yaw_threshold=18.0,
            pitch_threshold=20.0,
            iris_threshold=0.18,
            yaw_off_threshold=28.0,
            iris_off_threshold=0.28,
            window_size=12
        )
    
    def process(self, frame):
        """
        Run all processing steps on a single frame.
        - FaceMesh detection
        - Orientation estimation (2D-based)
        - Attention state update
        - Drawing
        
        :param frame: Input frame in BGR format (modified in-place for drawing).
        :return: A dictionary with processing results:
                 {"orientation": dict with yaw_score/pitch_score or None,
                  "attention": dict with state information}
        """
        # Detect face landmarks
        landmarks = self.face_detector.detect(frame)
        
        # Get frame dimensions
        frame_shape = (frame.shape[0], frame.shape[1])
        
        # Compute orientation from landmarks
        if landmarks is not None:
            orientation = self.orientation_estimator.compute(landmarks, frame_shape)
        else:
            orientation = None
        
        # Update attention state
        attention_info = self.attention_logic.update(orientation, timestamp=None)
        
        # Draw visualizations
        if landmarks is not None:
            # Draw face mesh on the frame (modifies frame in-place)
            draw_face_mesh(frame, landmarks)
        
        # Draw attention overlay with current state
        draw_attention_overlay(frame, orientation, attention_info["smoothed_state"])
        
        # Return results dictionary
        return {
            "orientation": orientation,
            "attention": attention_info
        }
    
    def close(self):
        """Release resources."""
        self.face_detector.close()

