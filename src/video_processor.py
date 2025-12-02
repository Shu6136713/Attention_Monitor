"""
Video Processor Module
High-level frame processing that orchestrates detection and visualization.
"""

from src.face_mesh_detector import FaceMeshDetector
from src.gaze_logic import OrientationEstimator, AttentionLogic, GazeState
from src.utils.visualizer import draw_face_mesh, draw_attention_overlay
from src.stats import StatsManager
from src.depth_estimator import DepthEstimator
from src.utils.mesh_visualizer import MeshVisualizer
from src.head_pose import HeadPoseEstimator


class VideoProcessor:
    """High-level processor that coordinates face detection and drawing."""
    
    def __init__(self, visualize_3d: bool = False):
        """
        Initialize the video processor with FaceMesh detector, orientation estimator, attention logic, and stats manager.
        
        :param visualize_3d: Whether to show a 3D plot of the face mesh.
        """
        self.face_detector = FaceMeshDetector()
        self.orientation_estimator = OrientationEstimator()
        self.head_pose_estimator = HeadPoseEstimator()
        self.attention_logic = AttentionLogic(
            yaw_threshold=20.0,
            pitch_threshold=15.0,
            iris_threshold=0.18,
            yaw_off_threshold=30.0,
            pitch_up_off_threshold=20.0,
            pitch_down_off_threshold=20.0,
            iris_off_threshold=0.28,
            window_size=5
        )
        self.stats_manager = StatsManager()
        
        # New modules for 3D estimation
        self.depth_estimator = DepthEstimator()
        self.visualize_3d = visualize_3d
        self.mesh_visualizer = MeshVisualizer() if visualize_3d else None
        
        # Calibration offsets (determined empirically based on user feedback)
        # User reported Raw Pitch ~ 36 when looking straight. Subtracting to center at 0.
        self.pitch_calibration_offset = -36.0
    
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
            # This overwrites the heuristic yaw/pitch with more accurate 3D model-based values
            pose = self.head_pose_estimator.estimate(landmarks, frame_shape)
            
            # DEBUG PRINTS
            # print(f"Landmarks detected. Orientation: {orientation is not None}, Pose: {pose is not None}")
            
            if orientation is not None and pose is not None:
                pnp_yaw, pnp_pitch, pnp_roll = pose
                
                # Apply calibration offset
                pnp_pitch += self.pitch_calibration_offset
                
                # DEBUG: Print raw values to understand why it fails
                print(f"DEBUG -> Yaw: {pnp_yaw:.1f}, Pitch: {pnp_pitch:.1f} (Raw: {pose[1]:.1f}), Iris: {orientation.get('iris_offset', 0):.3f}")
                
                orientation["yaw_angle"] = pnp_yaw
                orientation["pitch_angle"] = pnp_pitch
                orientation["roll_angle"] = pnp_roll
            elif orientation is not None:
                 # Fallback to heuristic if PnP fails but orientation succeeded
                 # print(f"Using heuristic angles -> Yaw: {orientation.get('yaw_angle'):.1f}")
                 pass
            
            # Estimate 3D points
            points_3d = self.depth_estimator.estimate_depth(landmarks, frame_shape)
            
            # Visualize in 3D if enabled
            if self.visualize_3d and self.mesh_visualizer:
                connections = self.depth_estimator.get_face_mesh_connections()
                self.mesh_visualizer.plot_3d_mesh(points_3d, connections)
                
        else:
            orientation = None
            points_3d = None
        
        # Update attention state
        attention_info = self.attention_logic.update(orientation, timestamp=None)
        
        # Update statistics with current gaze state
        self.stats_manager.update(attention_info["smoothed_state"])
        
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
    
    def get_stats_summary(self) -> dict:
        """
        Get attention statistics summary.
        
        :return: Dictionary with complete statistics including GA Rate and attention classification.
        """
        return self.stats_manager.get_summary()
    
    def close(self):
        """Release resources."""
        self.face_detector.close()
        if self.mesh_visualizer:
            self.mesh_visualizer.close()

