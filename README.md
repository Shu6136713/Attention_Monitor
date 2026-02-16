# Real-Time Gaze Detection System

## Overview
The Real-Time Gaze Detection System is a cutting-edge project that utilizes MediaPipe and OpenCV to classify attention levels based on webcam video input. This system allows for the assessment of a user's engagement by analyzing their gaze direction in real-time, making it an essential tool for applications in education, user experience research, and interaction design.

## Features
- **Real-Time Detection**: Detects and classifies gaze in real-time using efficient computer vision algorithms.
- **Attention Level Classification**: Provides insights into attention levels during activities done in front of the computer.
- **Robust Performance**: Built with MediaPipe and OpenCV ensuring high accuracy and performance.
- **Offline Functionality**: Includes an offline mode to process pre-recorded videos.

## Installation
To set up the project, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Shu6136713/Attention_Monitor.git
   cd Attention_Monitor
   ```
2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Real-Time Mode**:
   Run the main program using:
   ```bash
   python main.py
   ```
   This will open your webcam and begin real-time monitoring of attention levels.

2. **Offline Mode**:
   To analyze a pre-recorded video, use:
   ```bash
   python main_offline.py --video path_to_your_video.mp4
   ```
   Replace `path_to_your_video.mp4` with the actual path to your video file.

## Technical Details
- **Core Modules**:
  - `main.py`: The entry point for real-time gaze detection.
  - `main_offline.py`: Facilitates offline video analysis.
  - `src/VideoProcessor.py`: Responsible for handling video input and processing frames for gaze detection.
- **Technologies Used**: MediaPipe, OpenCV, Python
- **Additional Notes**: Ensure that your webcam is enabled and properly configured to utilize the real-time detection feature.