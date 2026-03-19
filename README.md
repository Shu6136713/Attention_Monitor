# Attention Monitor

## Comprehensive Documentation

### Overview
The **Attention Monitor** is a system designed to calculate yaw, pitch, and roll using the Perspective-n-Point (PnP) algorithm. This system leverages advanced attention logic classification and temporal smoothing techniques to enhance monitoring accuracy and reliability.

### Key Features
1. **Yaw, Pitch, and Roll Calculations**: The system accurately calculates the orientation of the monitored subject’s head using PnP algorithms. This involves deriving the angles from 3D landmarks.

2. **PnP Algorithm**: Using intrinsic camera parameters, this algorithm finds the location and orientation of objects in 3D. By inputting 2D image points and their corresponding 3D points, we can compute the camera pose.

3. **Attention Logic Classification**: This feature uses machine learning techniques to classify the attention state of the monitored subject, providing real-time insights into their attentiveness.

4. **Temporal Smoothing**: We apply temporal smoothing to minimize abrupt changes in direction and ensure stable readings by filtering out noise.

### Architecture
- **Input Module**: Captures video input from a camera.
- **Processing Module**: Implements PnP for calculations and runs the attention logic classification.
- **Output Module**: Displays results and statistics in real-time.

### Demo
![Demo GIF](link_to_your_demo_gif_here)  
*Insert the link to the GIF showing the demo of the Attention Monitor.*

### Installation
1. **Clone the Repository**:  
   `git clone https://github.com/Shu6136713/Attention_Monitor.git`
2. **Navigate to the Directory**:  
   `cd Attention_Monitor`
3. **Install Dependencies**:  
   Run the following command:  
   `pip install -r requirements.txt`

### How to Run
1. **Run the Application**:  
   After installing the necessary packages, execute the following command to start the application:  
   `python main.py`
2. **Follow the On-Screen Instructions**:  
   The application will guide you through the process of monitoring attention.

### Conclusion
The **Attention Monitor** is a powerful tool designed to provide real-time monitoring of attentiveness using innovative algorithms and stable calculations. Its comprehensive architecture and ease of use make it suitable for various applications, including in education and workplace settings.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.