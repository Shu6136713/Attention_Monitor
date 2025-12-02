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


def run_face_mesh_demo(video_path: Optional[str] = None, window_title: str = "Face Mesh Debug", visualize_3d: bool = False) -> None:
    """
    Run a Face Mesh demo using MediaPipe on a video file or webcam.
    
    :param video_path: Optional path to a video file. If None, use the webcam.
    :param window_title: Title for the OpenCV display window.
    :param visualize_3d: Whether to show 3D mesh visualization.
    """
    # Create video processor
    processor = VideoProcessor(visualize_3d=visualize_3d)
    
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
        # Display statistics summary before cleanup
        print("\n" + "="*60)
        print("Attention Analysis Summary")
        print("="*60)
        
        summary = processor.get_stats_summary()
        
        print(f"Total frames: {summary['total_frames']}")
        print(f"Frames with gaze on-screen: {summary['on_screen_frames']}")
        print(f"Frames with gaze aversion: {summary['off_screen_frames']}")
        print(f"Frames with unknown state: {summary['unknown_frames']}")
        print(f"\nGA Rate (Gaze Aversion Rate): {summary['ga_rate']:.2f}%")
        print(f"\nAttention Status: {summary['attention_label']}")
        
        # Add interpretation
        attention_level = summary['attention_level']
        print("\nInterpretation:")
        if attention_level.name == "FOCUSED":
            print("  User was highly focused on task. Low cognitive load.")
        elif attention_level.name == "NORMAL":
            print("  User had normal attention. Occasional gaze aversion for thinking.")
        else:  # DISTRACTED
            print("  User was distracted. High cognitive load or discomfort.")
        
        print("="*60 + "\n")
        
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
    show_3d = False
    
    args = sys.argv[1:]
    if "--3d" in args:
        show_3d = True
        args.remove("--3d")
    
    if len(args) > 0:
        video_file = args[0]
    
    run_face_mesh_demo(video_path=video_file, visualize_3d=show_3d)
