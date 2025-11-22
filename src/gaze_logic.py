"""
Gaze Logic Module
Contains orientation estimation and attention state management based on 2D landmarks.
"""

import numpy as np
from enum import Enum, auto
from typing import Optional, Dict
from collections import deque


class GazeState(Enum):
    """Enumeration of possible gaze states."""
    UNKNOWN = auto()
    ON_SCREEN = auto()
    OFF_SCREEN = auto()


class OrientationEstimator:
    """
    Estimate head and gaze orientation using geometry-based vector relationships.

    This approach aims to be largely invariant to distance from camera and
    head position changes. It uses internal facial geometry (ear-to-ear, eye
    direction, iris position) to compute stable yaw and pitch angles.
    """

    def __init__(self):
        """Initialize the orientation estimator with landmark indices."""
        # Landmark indices for MediaPipe FaceMesh (468 points)
        # Ears
        self.left_ear_idx = 234
        self.right_ear_idx = 454

        # Nose and chin
        self.nose_tip_idx = 1
        self.chin_idx = 152

        # Eyes
        self.left_eye_inner_idx = 133
        self.left_eye_outer_idx = 33
        self.right_eye_inner_idx = 362
        self.right_eye_outer_idx = 263

        # Iris centers (available with refine_landmarks=True)
        self.left_iris_center_idx = 468
        self.right_iris_center_idx = 473

        # Thresholds for classification (used by AttentionLogic)
        self.yaw_on_threshold = 18.0      # degrees
        self.pitch_on_threshold = 20.0    # degrees
        self.iris_on_threshold = 0.18     # normalized
        self.yaw_off_threshold = 28.0     # degrees
        self.iris_off_threshold = 0.28    # normalized

    def _get_point(self, landmarks, idx: int) -> Optional[np.ndarray]:
        """
        Extract a 2D point from landmarks (normalized coordinates).

        :param landmarks: MediaPipe face_landmarks object.
        :param idx: Landmark index.
        :return: numpy array [x, y] in normalized coordinates (0..1), or None if invalid.
        """
        try:
            lm = landmarks.landmark[idx]
            return np.array([lm.x, lm.y], dtype=np.float64)
        except (IndexError, AttributeError):
            return None

    def _normalize(self, vector: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize a vector to unit length.

        :param vector: Input vector.
        :return: Normalized vector, or None if zero-length.
        """
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return None
        return vector / norm

    def _signed_angle(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute signed angle between two 2D vectors in degrees.

        Uses atan2 for proper quadrant handling. Positive angle means
        counter-clockwise rotation from a to b.

        :param a: First vector (numpy array).
        :param b: Second vector (numpy array).
        :return: Signed angle in degrees in range [-180, 180].
        """
        # Compute the cross product (z-component in 2D: a.x * b.y - a.y * b.x)
        cross = a[0] * b[1] - a[1] * b[0]

        # Compute the dot product
        dot = float(np.dot(a, b))

        # Use atan2 to get signed angle
        angle_rad = np.arctan2(cross, dot)

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    def _minimal_signed_angle(self, angle_deg: float) -> float:
        """
        Map an angle in degrees to the minimal signed representation in [-90, 90].

        This helps to fix cases where we get ~180° because vector direction
        was flipped (e.g. true difference is 0°, but vectors are reversed).

        :param angle_deg: Input angle in degrees (any range).
        :return: Angle in range [-90, 90] representing the smallest equivalent rotation.
        """
        # First wrap to [-180, 180]
        angle = (angle_deg + 180.0) % 360.0 - 180.0

        # Then fold into [-90, 90]
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0

        return float(angle)

    def _horizontal_vector(self, p1: np.ndarray, p2: np.ndarray) -> Optional[np.ndarray]:
        """
        Build a horizontal vector that always points to the right in image space.

        This avoids sign flips where the anatomical "inner" / "outer" labels
        do not match left/right in the image (mirroring, camera orientation, etc).

        :param p1: First point [x, y].
        :param p2: Second point [x, y].
        :return: Normalized 2D vector pointing roughly to the right, or None.
        """
        if p1 is None or p2 is None:
            return None

        # We want a vector that points from the leftmost point to the rightmost.
        if p1[0] <= p2[0]:
            v = p2 - p1
        else:
            v = p1 - p2

        return self._normalize(v)

    def compute(self, landmarks, frame_shape: tuple) -> Optional[Dict]:
        """
        Compute orientation and gaze information from facial landmarks.

        :param landmarks: MediaPipe face_landmarks (468 points).
        :param frame_shape: Tuple of (height, width) of the frame (not used, but kept for API).
        :return: Dictionary with:
                 - 'yaw_angle': float (degrees, ~0 when looking straight)
                 - 'pitch_angle': float (degrees, ~0 when head upright)
                 - 'iris_offset': float (normalized horizontal, ~0 when eyes centered)
                 or None if landmarks are invalid.
        """
        if landmarks is None:
            return None

        try:
            # ===== STEP 1: Extract Required Landmarks =====
            left_ear = self._get_point(landmarks, self.left_ear_idx)
            right_ear = self._get_point(landmarks, self.right_ear_idx)
            nose_tip = self._get_point(landmarks, self.nose_tip_idx)
            chin = self._get_point(landmarks, self.chin_idx)

            left_eye_inner = self._get_point(landmarks, self.left_eye_inner_idx)
            left_eye_outer = self._get_point(landmarks, self.left_eye_outer_idx)
            right_eye_inner = self._get_point(landmarks, self.right_eye_inner_idx)
            right_eye_outer = self._get_point(landmarks, self.right_eye_outer_idx)

            left_iris_center = self._get_point(landmarks, self.left_iris_center_idx)
            right_iris_center = self._get_point(landmarks, self.right_iris_center_idx)

            # Check if all required points are valid
            required_points = [
                left_ear, right_ear, nose_tip, chin,
                left_eye_inner, left_eye_outer, right_eye_inner, right_eye_outer,
                left_iris_center, right_iris_center,
            ]
            if any(p is None for p in required_points):
                return None

            # ===== STEP 2: Build Normalized Vectors =====

            # a. Ear-to-ear horizontal head direction (always pointing right in image)
            # Determine which ear is left/right in image coordinates by x position
            if left_ear[0] <= right_ear[0]:
                # left_ear really on the left side of the image
                E_raw = right_ear - left_ear
            else:
                # Swap if indexing and image orientation disagree
                E_raw = left_ear - right_ear

            E = self._normalize(E_raw)
            if E is None:
                return None

            # b. Nose-to-chin vertical head direction
            V_raw = chin - nose_tip
            V = self._normalize(V_raw)
            if V is None:
                return None

            # c. Eye direction (both eyes aligned to point right in image)
            LE = self._horizontal_vector(left_eye_inner, left_eye_outer)
            RE = self._horizontal_vector(right_eye_inner, right_eye_outer)

            if LE is None or RE is None:
                return None

            # Average eye direction
            eye_dir_raw = (LE + RE) / 2.0
            eye_dir = self._normalize(eye_dir_raw)
            if eye_dir is None:
                return None

            # d. Iris displacement (horizontal offset inside each eye)
            left_eye_center = (left_eye_inner + left_eye_outer) / 2.0
            right_eye_center = (right_eye_inner + right_eye_outer) / 2.0

            # Compute normalized eye widths (inner↔outer)
            left_eye_width = abs(left_eye_outer[0] - left_eye_inner[0])
            right_eye_width = abs(right_eye_outer[0] - right_eye_inner[0])

            # Prevent division by zero
            if left_eye_width < 1e-6 or right_eye_width < 1e-6:
                return None

            # Raw displacement
            iris_offset_left_raw = left_iris_center - left_eye_center
            iris_offset_right_raw = right_iris_center - right_eye_center

            # Normalize displacement by eye width
            iris_offset_left = iris_offset_left_raw[0] / left_eye_width
            iris_offset_right = iris_offset_right_raw[0] / right_eye_width

            # Average eyes
            iris_offset_avg_x = float((iris_offset_left + iris_offset_right) / 2.0)

            # ===== STEP 3: Compute Signed Angles =====

            # Yaw: rotation of eye direction relative to head horizontal axis.
            # When looking straight, eye_dir and E should be almost aligned → ~0°.
            yaw_raw = self._signed_angle(eye_dir, E)
            yaw_angle = self._minimal_signed_angle(yaw_raw)

            # Pitch: rotation of nose→chin vector relative to global vertical axis.
            # Global vertical axis points "down" in image coordinates: [0, 1].
            vertical_axis = np.array([0.0, 1.0], dtype=np.float64)
            pitch_raw = self._signed_angle(V, vertical_axis)
            pitch_angle = self._minimal_signed_angle(pitch_raw)

            return {
                "yaw_angle": yaw_angle,
                "pitch_angle": pitch_angle,
                "iris_offset": iris_offset_avg_x,
            }

        except (IndexError, AttributeError, ValueError):
            # Invalid landmarks structure or computation error
            return None


class AttentionLogic:
    """
    Map orientation angles and iris offset to a stable ON_SCREEN/OFF_SCREEN state,
    with simple temporal smoothing and event counting.
    """

class AttentionLogic:
    def __init__(
        self,
        # === Balanced thresholds ===
        yaw_threshold: float = 12.0,         # ON עד 12° לכל צד
        pitch_threshold: float = 15.0,       # ON עד 15°
        iris_threshold: float = 0.035,       # ON – סטייה קטנה של העיניים

        yaw_off_threshold: float = 21.0,     # OFF – הראש באמת יוצא הצידה
        iris_off_threshold: float = 0.055,   # OFF – העיניים באמת בורחות

        # === Balanced smoothing ===
        window_size: int = 8,                # 8 פריימים היסטוריה
    ):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.iris_threshold = iris_threshold
        self.yaw_off_threshold = yaw_off_threshold
        self.iris_off_threshold = iris_off_threshold
        self.window_size = window_size

        self.state_history = deque(maxlen=window_size)
        self.smoothed_state = GazeState.UNKNOWN
        self.last_smoothed_state = GazeState.UNKNOWN
        self.off_events_count = 0


    def update(
        self,
        orientation: Optional[Dict],
        timestamp: Optional[float] = None,
    ) -> Dict:
        """
        Update attention state based on current orientation.

        :param orientation: Dict with 'yaw_angle', 'pitch_angle', and 'iris_offset' (or None).
        :param timestamp: Optional time in seconds (can be frame index or real time).
        :return: Dict with:
            - "raw_state": GazeState
            - "smoothed_state": GazeState
            - "just_switched_on": bool
            - "just_switched_off": bool
            - "off_events_count": int
        """
        # Determine raw state based on orientation
        if orientation is None:
            raw_state = GazeState.UNKNOWN
        else:
            yaw_angle = float(orientation.get("yaw_angle", 0.0))
            pitch_angle = float(orientation.get("pitch_angle", 0.0))
            iris_offset = float(orientation.get("iris_offset", 0.0))

            # Check for ON_SCREEN condition (all must be true)
            if (
                abs(yaw_angle) < self.yaw_threshold
                and abs(pitch_angle) < self.pitch_threshold
                and abs(iris_offset) < self.iris_threshold
            ):
                raw_state = GazeState.ON_SCREEN

            # Check for OFF_SCREEN condition (any can be true)
            elif (
                abs(yaw_angle) > self.yaw_off_threshold
                or abs(iris_offset) > self.iris_off_threshold
            ):
                raw_state = GazeState.OFF_SCREEN

            # Otherwise, it's in the uncertain zone
            else:
                raw_state = GazeState.UNKNOWN

        # Add raw state to history
        self.state_history.append(raw_state)

        # Compute smoothed state by majority vote
        on_count = sum(1 for s in self.state_history if s == GazeState.ON_SCREEN)
        off_count = sum(1 for s in self.state_history if s == GazeState.OFF_SCREEN)
        unknown_count = sum(1 for s in self.state_history if s == GazeState.UNKNOWN)

        # Determine smoothed state (majority wins)
        if on_count > off_count and on_count > unknown_count:
            new_smoothed_state = GazeState.ON_SCREEN
        elif off_count > on_count and off_count > unknown_count:
            new_smoothed_state = GazeState.OFF_SCREEN
        else:
            # Tie or mostly unknown - keep previous state or default to UNKNOWN
            new_smoothed_state = (
                self.smoothed_state
                if self.smoothed_state != GazeState.UNKNOWN
                else GazeState.UNKNOWN
            )

        # Detect transitions
        just_switched_on = False
        just_switched_off = False

        if (
            self.last_smoothed_state == GazeState.OFF_SCREEN
            and new_smoothed_state == GazeState.ON_SCREEN
        ):
            just_switched_on = True

        if (
            self.last_smoothed_state == GazeState.ON_SCREEN
            and new_smoothed_state == GazeState.OFF_SCREEN
        ):
            just_switched_off = True
            self.off_events_count += 1

        # Update state tracking
        self.last_smoothed_state = self.smoothed_state
        self.smoothed_state = new_smoothed_state

        return {
            "raw_state": raw_state,
            "smoothed_state": self.smoothed_state,
            "just_switched_on": just_switched_on,
            "just_switched_off": just_switched_off,
            "off_events_count": self.off_events_count,
        }
