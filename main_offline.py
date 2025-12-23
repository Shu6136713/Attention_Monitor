"""
Face Mesh Demo using MediaPipe
This script demonstrates facial landmark detection using MediaPipe Face Mesh.
It can work with either a video file or webcam feed.
"""

import sys
from typing import Optional
import cv2

from src.video_processor import VideoProcessor


def get_video_capture(video_path: Optional[str] = None) -> cv2.VideoCapture:
    """
    Create and return a VideoCapture object.
    If video_path is provided, open that file.
    Otherwise, open the default webcam (device 0).
    
    :param video_path: Optional path to a video file.
    :return: cv2.VideoCapture instance.
    :raises RuntimeError: if the capture source cannot be opened.
    """
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam (device 0)")
    
    return cap


def run_face_mesh_demo(video_path: Optional[str] = None, window_title: str = "Face Mesh Debug") -> None:
    """
    Run a Face Mesh demo using MediaPipe on a video file or webcam.
    
    :param video_path: Optional path to a video file. If None, use the webcam.
    :param window_title: Title for the OpenCV display window.
    """
    # Create video processor
    processor = VideoProcessor()
    
    try:
        # Get video capture object (either from file or webcam)
        cap = get_video_capture(video_path)
        
        print(f"Processing video... Press 'q' to quit.")
        if video_path:
            print(f"Video source: {video_path}")
        else:
            print("Video source: Webcam (device 0)")
        
        # Main processing loop
        while True:
            # Read frame from video capture
            ret, frame = cap.read()
            
            # Check if frame was successfully read
            if not ret:
                print("End of video stream or failed to read frame.")
                break
            
            # Process the frame (detection + drawing)
            output = processor.process(frame)
            
            # Temporary debug: print orientation and attention info
            print(f"Orientation: {output.get('orientation')}")
            print(f"Attention: {output.get('attention')}")
            print("---")
            
            # Display the frame with face mesh overlay
            cv2.imshow(window_title, frame)
            
            # Check if user pressed 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit.")
                break
    
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    
    finally:
        # Clean up resources
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        processor.close()
        print("Resources released. Goodbye!")


if __name__ == "__main__":
    # If a video path is passed as the first argument, use it.
    # Otherwise, fall back to the webcam.
    video_file = None
    
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    
    run_face_mesh_demo(video_path=video_file)
