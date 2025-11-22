# Face Mesh Demo with MediaPipe

A Python project that demonstrates facial landmark detection using MediaPipe Face Mesh on video files or webcam feed.

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
uv run python attention_monitor/main_offline.py
```

### Running with a video file

```bash
uv run python attention_monitor/main_offline.py path/to/video.mp4
```

For example, to use one of the sample videos:

```bash
uv run python attention_monitor/main_offline.py attention_monitor/data/videos/WIN_20251119_10_48_47_Pro.mp4
```

## Controls

- Press **'q'** to quit the application

## Features

- Real-time face mesh detection using MediaPipe
- Works with both video files and webcam
- Draws face mesh tessellation, contours, and iris landmarks
- Graceful error handling
- Clean resource management

## Requirements

- Python 3.9+
- opencv-python
- mediapipe

