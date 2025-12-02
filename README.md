# Gaze Attention Tracker

A system for monitoring attention and tracking gaze direction using MediaPipe face detection. The system detects when the user is looking at the screen and when they avert their gaze, calculating attention metrics (GA Rate) and attention level classification.

## Current State

The project is **fully functional and ready to use** with the following features:

### ✅ What Works Now:
- **Real-time face detection** - Uses MediaPipe Face Mesh to detect 468 facial landmarks
- **Head and gaze orientation estimation** - Calculates yaw and pitch based on facial geometry
- **Gaze state detection** - Automatically detects if gaze is on-screen (ON_SCREEN) or off-screen (OFF_SCREEN)
- **GA Rate calculation** - Calculates Gaze Aversion Rate (percentage of time gaze is off-screen)
- **Attention level classification** - Automatic classification into 3 levels:
  - **Focused** (< 10% GA Rate) - User is highly focused on task
  - **Normal** (10-20% GA Rate) - User occasionally looks away for thinking
  - **Distracted** (> 20% GA Rate) - High cognitive load or discomfort
- **Visualization** - Draws face mesh and attention state information on screen
- **Video processing** - Supports processing video files or webcam feed
- **Statistics summary** - Displays a complete analysis summary at the end of execution

### 📊 Technical Infrastructure:

**Main Technologies:**
- **MediaPipe Face Mesh** - Detects 468 facial landmarks
- **OpenCV** - Image and video processing
- **NumPy** - Mathematical and geometric calculations
- **Python 3.9+** - Programming language

**Project Structure:**
- `main_offline.py` - Main entry point
- `src/video_processor.py` - Central video processor that coordinates all modules
- `src/face_mesh_detector.py` - Face detection with MediaPipe
- `src/gaze_logic.py` - Head and gaze orientation estimation + attention logic
- `src/stats.py` - Statistics management and GA Rate calculation
- `src/utils/visualizer.py` - Visualization functions
- `src/head_pose.py` - Additional module for head pose estimation

**Algorithm:**
The system uses a geometry-based approach to calculate head and gaze orientation:
- Yaw calculation (left-right rotation) based on distance between ears
- Pitch calculation (up-down rotation) based on nose and chin position
- Iris position analysis relative to eye center
- Window smoothing for result stabilization

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

Or install the required packages directly:

```bash
uv pip install opencv-python mediapipe
```

## Usage

### Running with webcam (default)

```bash
uv run python main_offline.py
```

Or:

```bash
python main_offline.py
```

### Running with a video file

```bash
uv run python main_offline.py path/to/video.mp4
```

Example using one of the existing video files:

```bash
python main_offline.py data/videos/WIN_20251119_10_48_47_Pro.mp4
```

## Controls

- Press **'q'** to quit the application

## Features

- **Real-time face detection** - Detects 468 facial landmarks with MediaPipe Face Mesh
- **Video or webcam processing** - Supports processing video files or webcam feed
- **Head and gaze orientation estimation** - Calculates yaw and pitch based on facial geometry
- **Gaze state detection** - Automatically detects if gaze is on-screen or off-screen
- **GA Rate calculation** - Calculates Gaze Aversion Rate (percentage of time gaze is off-screen)
- **Attention level classification** - Automatic classification into 3 levels: Focused, Normal, Distracted
- **Visualization** - Draws face mesh, contours, and iris landmarks on screen
- **Statistics summary** - Displays a complete analysis summary at the end of execution
- **Resource management** - Proper resource cleanup on exit

## Requirements

- Python 3.9+
- opencv-python >= 4.8.0
- mediapipe >= 0.10.0
- numpy (installed automatically with mediapipe)

## Output

At the end of execution, the system displays a complete summary including:
- Total number of frames
- Number of frames with gaze on-screen
- Number of frames with gaze off-screen
- GA Rate (percentage)
- Attention level (Focused/Normal/Distracted)
- Interpretation of the results

