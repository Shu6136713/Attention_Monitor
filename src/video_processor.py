"""
Video Processor Module
High-level frame processing that orchestrates detection and visualization.
"""

from src.face_mesh_detector import FaceMeshDetector
from src.gaze_logic import OrientationEstimator, AttentionLogic, GazeState
from src.utils.visualizer import draw_face_mesh, draw_attention_overlay
from src.head_pose import HeadPoseEstimator  # Re-import HeadPoseEstimator


class VideoProcessor:
    """High-level processor that coordinates face detection and drawing."""
    
    def __init__(self):
        """Initialize the video processor with FaceMesh detector, orientation estimator, and attention logic."""
        self.face_detector = FaceMeshDetector()
        self.orientation_estimator = OrientationEstimator()
        self.head_pose_estimator = HeadPoseEstimator()
        
        # TIGHTER Thresholds for high sensitivity
        self.attention_logic = AttentionLogic(
            yaw_threshold=12.0,           # Sensitive: 12 degrees deviation allowed
            pitch_threshold=10.0,         # Sensitive: 10 degrees deviation allowed
            iris_threshold=0.15,          # Strict: Eyes must be very centered
            
            yaw_off_threshold=20.0,       # OFF if > 20 degrees
            pitch_up_off_threshold=15.0,  # OFF if looking up > 15 degrees
            pitch_down_off_threshold=15.0,# OFF if looking down > 15 degrees
            iris_off_threshold=0.18,      # OFF if eyes move > 0.18 (catches squints at ~0.19)
            
            window_size=5                 # Fast response
        )
        
        # Calibration offset for HeadPoseEstimator
        # With the original model, raw pitch is around +37 when looking straight
        # So we add +37 to center it at 0
        self.pitch_calibration_offset = 37.0
    
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
            # Get basic orientation and iris state
            orientation = self.orientation_estimator.compute(landmarks, frame_shape)
            
            # Refine head pose using PnP (HeadPoseEstimator)
            pose = self.head_pose_estimator.estimate(landmarks, frame_shape)
            
            if orientation is not None and pose is not None:
                pnp_yaw, pnp_pitch, pnp_roll = pose
                
                # Apply calibration offset
                pnp_pitch += self.pitch_calibration_offset
                
                # Fix Yaw flip: if face is detected as "looking backwards" (around 180/-180),
                # flip it to front (around 0).
                if abs(pnp_yaw) > 160:
                    if pnp_yaw > 0:
                        pnp_yaw -= 180
                    else:
                        pnp_yaw += 180
                
                # DEBUG: Print raw values to monitor sensitivity
                print(f"DEBUG -> Yaw: {pnp_yaw:.1f}, Pitch: {pnp_pitch:.1f}, Iris: {orientation.get('iris_offset', 0):.3f}")
                
                orientation["yaw_angle"] = pnp_yaw
                orientation["pitch_angle"] = pnp_pitch
                orientation["roll_angle"] = pnp_roll
                
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

