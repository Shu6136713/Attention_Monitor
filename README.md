# Attention Monitor

## Project Overview
This project implements a real-time **Gaze Detection System**. It leverages advanced computer vision techniques to track and analyze gaze movement, making it invaluable for applications in various fields such as user experience testing, marketing analysis, and accessibility solutions.

## Idea Behind the Project
The primary goal of the Attention Monitor is to determine where a user is looking in real-time. By integrating face mesh technology, we aim to provide insights into attention distribution, enabling further improvements in design and content delivery.

## Technology Stack
- **Python 3.11**: The core programming language for developing the application.
- **MediaPipe Face Mesh**: For detecting and tracking facial landmarks.
- **OpenCV**: Utilized for image processing and manipulation.
- **NumPy**: Helps with numerical operations and handling arrays.

## Architecture
The architecture of the Attention Monitor consists of the following core modules:
- **FaceMeshDetector**: Responsible for detecting the face and extracting mesh points.
- **OrientationEstimator**: Analyzes the orientation of the user's face based on the detected mesh.
- **HeadPoseEstimator**: Estimates head pose angles to infer gaze direction.
- **AttentionLogic**: Integrates data from various modules to make decisions on attention metrics.
- **VideoProcessor**: Handles video streams (both live and recorded) for real-time processing.

## Installation
To get started with the project, install the necessary dependencies using the `uv` package manager:
```bash
uv install mediapipe opencv-python numpy
```

## Running the Application
You can run the application using either of the following:
1. **Main Application (Online Demo)**:  Execute `main.py` to start the online demo.
2. **Offline Mode**: Execute `main_offline.py` to run with a webcam or video file.

## Configuration and Threshold Settings
Configuration settings can be adjusted within the code to calibrate the gaze detection metrics. It's recommended to review comments in the code for optimal threshold selection based on user needs.

## Demo
Check out our online demo and visualize the system in action:
- [Demo Video](https://link-to-demo-video.com)

![Demo GIF](https://link-to-demo-gif.com/demo.gif)  

## Conclusion
The Attention Monitor project provides a robust foundation for understanding and analyzing gaze direction in real time. By utilizing modern technologies, it opens doors for improved engagement strategies in numerous domains.
