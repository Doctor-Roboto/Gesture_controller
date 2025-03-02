# Underwater Robot Gesture & Trajectory Control Package

## Overview

This package provides a modular framework for controlling an underwater robot using hand gestures. It integrates gesture recognition and trajectory planning (both 2D and 3D) using MediaPipe and OpenCV. The system supports multiple modes with real-time switching and includes an optional data collection mode for analysis.

## Getting started

- **If Using Franka PR3**
  - Launching basic controller with mode switcher
  - Currently <robot_host_name> is 192.168.1.11
```bash
python3 mode_manager.py <robot_host_name>
```
  - Initialisation of basic gestures for designated TAMP algorithm.
```bash
python3 gesturePR3.py <robot_host_name>
```

## Features

- **Gesture Recognition**
  - Recognizes gestures such as **stop**, **okay**, **go up**, **go down**, and **pointing spatial**.
  - Uses a confirmation mechanism (a gesture must be detected for a fixed number of frames) for reliable control.

- **Trajectory Planning**
  - **2D Trajectory Mode:** Collects and draws a 2D trajectory based on hand landmarks.
  - **3D Trajectory Mode:** Collects 3D trajectory data (using a scaled depth component) and displays a color-coded 2D projection indicating motion in and out of the screen.

- **Mode Switching**
  - Switch between gesture mode and trajectory mode using simple number gestures:
    - **2** (index and middle fingers extended) to select gesture mode.
    - **3** (index, middle, and ring fingers extended) to select trajectory mode.
  - Within trajectory mode, toggle between 2D and 3D sub-modes using the same gesture.
  - Confirmation of mode switches is achieved via the **okay** (confirm) and **stop** (cancel) gestures.

- **Data Collection (Optional)**
  - A separate launcher records 2D and 3D trajectory sessions as well as gesture events with timestamps.
  - Generates plots (using Matplotlib) for each trajectory session and a gesture summary for post-session analysis.

## File Structure

- **gestures.py**  
  Contains classes for individual gesture detection:
  - `StopGesture`, `OkayGesture`, `GoUpGesture`, `GoDownGesture`, and `PointingSpatialGesture`
  - `GestureRecognizer` aggregates these detectors and implements a confirmation mechanism.

- **trajectory.py**  
  Implements `TrajectoryMode` which samples hand landmarks to record and draw a 2D trajectory.

- **trajectory_3D.py**  
  Implements `Trajectory3DMode` which records 3D trajectory data (using the z coordinate) and draws a 2D projection with color cues for depth.

- **mode_manager.py**  
  The main launcher that integrates gesture and trajectory modes. It handles mode switching and displays the current mode on the video feed.

- **data_collection.py**  
  A variant of the launcher that logs trajectory sessions and gesture events. At the end of a session, it produces plots for 2D trajectories, 3D trajectories, and a gesture summary.

## Requirements

The package relies on the following key dependencies:

- **numpy**
- **mediapipe**
- **opencv-python**
- **matplotlib**
- **panda_py** (for robot control)

Install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
