import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class MeshVisualizer:
    """
    Helper class to visualize 3D face mesh.
    """
    
    def __init__(self):
        self.fig = None
        self.ax = None

    def plot_3d_mesh(self, points_3d, connections=None, title="3D Face Mesh"):
        """
        Create a 3D plot of the face mesh.
        
        :param points_3d: Numpy array of shape (N, 3).
        :param connections: Optional set of connections (start_idx, end_idx).
        """
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax.clear()

        # Extract X, Y, Z
        # Note: In image coords, Y is down. In 3D plots, usually Z is up.
        # We map: 
        # X -> X
        # Y -> -Y (to flip image coordinate system to standard cartesian)
        # Z -> Z (depth)
        
        xs = points_3d[:, 0]
        ys = -points_3d[:, 1] # Flip Y
        zs = points_3d[:, 2]

        self.ax.scatter(xs, zs, ys, c='b', marker='.', s=1, alpha=0.5)

        if connections:
            # Plot a subset of connections to avoid being too slow
            # Convert connections to list if it's a frozen set
            conn_list = list(connections)
            # Limit number of lines for performance if needed, or plot all
            for start, end in conn_list[::2]: # Plot every 2nd line for speed
                p1 = points_3d[start]
                p2 = points_3d[end]
                self.ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], 'k-', linewidth=0.2, alpha=0.3)

        self.ax.set_xlabel('X (Pixels)')
        self.ax.set_ylabel('Z (Depth)')
        self.ax.set_zlabel('Y (Height)')
        self.ax.set_title(title)
        
        # Set equal aspect ratio logic (approximation)
        self.ax.set_box_aspect([1,1,1])
        
        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)

