import numpy as np
import mediapipe as mp

class DepthEstimator:
    """
    Estimates 3D depth (Z) for facial landmarks based on 2D positions and relative face geometry.
    """

    def __init__(self, focal_length_scale: float = 1.0, avg_face_width_cm: float = 14.0):
        """
        Initialize the Depth Estimator.

        :param focal_length_scale: Multiplier for estimated focal length (relative to image width).
        :param avg_face_width_cm: Average real-world face width in centimeters.
        """
        self.focal_length_scale = focal_length_scale
        self.avg_face_width_cm = avg_face_width_cm
        
        # Key landmarks for face width estimation (Cheekbones/Ears area)
        # 454: Left ear tragus, 234: Right ear tragus (approximate outer points)
        self.LEFT_FACE_EDGE = 454
        self.RIGHT_FACE_EDGE = 234

    def estimate_depth(self, landmarks, image_shape):
        """
        Convert normalized landmarks to 3D coordinates with estimated absolute depth.

        :param landmarks: MediaPipe NormalizedLandmarkList.
        :param image_shape: Tuple (height, width).
        :return: Numpy array of shape (N, 3) containing (X, Y, Z) coordinates.
        """
        height, width = image_shape[:2]
        points_3d = []

        # 1. Convert all landmarks to pixel coordinates (2D) and extract relative Z
        # MediaPipe Z is relative to the face center of mass, roughly same scale as X.
        for lm in landmarks.landmark:
            px = lm.x * width
            py = lm.y * height
            # We use the provided relative Z, scaled by width to match pixel units
            # This covers the "Relative positions" requirement (nose vs ears)
            pz_rel = lm.z * width 
            points_3d.append([px, py, pz_rel])
        
        points_3d = np.array(points_3d)

        # 2. Estimate Global Depth (Z) based on Face Size
        # Calculate face width in pixels
        left_pt = points_3d[self.LEFT_FACE_EDGE]
        right_pt = points_3d[self.RIGHT_FACE_EDGE]
        
        # Euclidean distance in 2D (ignoring relative Z for width measurement)
        face_width_px = np.linalg.norm(left_pt[:2] - right_pt[:2])
        
        if face_width_px == 0:
            return points_3d # Avoid division by zero

        # Estimate focal length (simple heuristic: f = width for standard FOV)
        focal_length = width * self.focal_length_scale

        # Perspective projection: Z = (f * real_width) / pixel_width
        z_global = (focal_length * self.avg_face_width_cm) / face_width_px
        
        # However, since we want the result in "pixel-consistent" units or "camera space",
        # we can keep it in arbitrary units or scale it. 
        # Let's treat the units as 'pixels' for X,Y. For Z, we add the global offset.
        # To make Z comparable to X,Y (pixels), z_global is effectively the distance in "pixels" 
        # if we assume 1 pixel = 1 unit at the image plane.
        
        # Add global Z to relative Z
        # We assume the camera is at (width/2, height/2, 0) or similar, and face is at -Z or +Z.
        # Standard convention: Z increases away from camera.
        
        points_3d[:, 2] += z_global

        return points_3d

    def get_face_mesh_connections(self):
        """
        Return the list of connections (edges) for the face mesh topology.
        """
        return mp.solutions.face_mesh.FACEMESH_TESSELATION

