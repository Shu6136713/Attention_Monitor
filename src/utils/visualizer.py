"""
Visualizer Module
Contains drawing functions for visualizing face mesh landmarks and attention state.
"""

import cv2
import mediapipe as mp


# Initialize MediaPipe drawing utilities (module-level constants)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def draw_face_mesh(frame_bgr, landmarks):
    """
    Draw tessellation, contours, and irises on the frame.
    
    :param frame_bgr: Input frame in BGR format (modified in-place).
    :param landmarks: Face landmarks from MediaPipe FaceMesh detection.
    """
    if landmarks is None:
        return
    
    # Draw the face mesh tessellation
    mp_drawing.draw_landmarks(
        image=frame_bgr,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    
    # Draw the face mesh contours
    mp_drawing.draw_landmarks(
        image=frame_bgr,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    
    # Draw iris landmarks (available with refine_landmarks=True)
    mp_drawing.draw_landmarks(
        image=frame_bgr,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
    )


def draw_attention_overlay(frame, orientation, gaze_state):
    """
    Draw a minimal debug overlay showing attention state.
    
    Displays:
        - Current gaze_state (ON_SCREEN/OFF_SCREEN/UNKNOWN) as text
        - Yaw and pitch angles in degrees (if available)
        - Iris offset (if available)
    
    :param frame: Input frame in BGR format (modified in-place).
    :param orientation: Dictionary with 'yaw_angle', 'pitch_angle', and 'iris_offset' (or None).
    :param gaze_state: GazeState enum value (ON_SCREEN/OFF_SCREEN/UNKNOWN).
    """
    # Import GazeState here to avoid circular imports
    from src.gaze_logic import GazeState
    
    # Position for text overlay (top-left corner)
    x_pos = 10
    y_pos = 30
    line_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Choose color based on gaze state
    if gaze_state == GazeState.ON_SCREEN:
        color = (0, 255, 0)  # Green for ON_SCREEN
        state_text = "State: ON_SCREEN"
    elif gaze_state == GazeState.OFF_SCREEN:
        color = (0, 0, 255)  # Red for OFF_SCREEN
        state_text = "State: OFF_SCREEN"
    else:
        color = (128, 128, 128)  # Gray for UNKNOWN
        state_text = "State: UNKNOWN"
    
    # Draw state text
    cv2.putText(frame, state_text, (x_pos, y_pos), font, font_scale, color, thickness)
    
    # Draw orientation information if available
    if orientation is not None:
        yaw_angle = orientation.get("yaw_angle", 0.0)
        pitch_angle = orientation.get("pitch_angle", 0.0)
        iris_offset = orientation.get("iris_offset", 0.0)
        
        yaw_text = f"Yaw: {yaw_angle:+.1f} deg"
        pitch_text = f"Pitch: {pitch_angle:+.1f} deg"
        iris_text = f"Iris: {iris_offset:+.3f}"
        
        cv2.putText(frame, yaw_text, (x_pos, y_pos + line_height), 
                   font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, pitch_text, (x_pos, y_pos + 2 * line_height), 
                   font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, iris_text, (x_pos, y_pos + 3 * line_height), 
                   font, font_scale, (255, 255, 255), thickness)
