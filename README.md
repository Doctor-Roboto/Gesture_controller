# Robot Gesture & Trajectory Control

## Overview

This package provides a modular framework for controlling an underwater robot using hand gestures. It integrates gesture recognition and trajectory planning (both 2D and 3D) using MediaPipe and OpenCV. The system supports multiple modes with real-time switching and includes an optional data collection mode for analysis.

## Getting started

- **Set up environment(skip if using Franka PR3)**
  - Create and activate virtual environment.
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
- Install the package 
```bash
pip install git+https://github.com/Doctor-Roboto/Gesture_controller.git
```
- **Launch gesture controller using terminal**
    - Choose between standard controller `mode_manager.py` or recording controller `data_collection.py` and launch. 
```bash
python3 gesture/mode_manager.py
```
```bash
python3 gesture/data_collection.py
```

  
## If Using Franka PR3
- **Ensure you have navigated to the proper directory and install**
```bash
pip install git+https://github.com/Doctor-Roboto/Gesture_controller.git
```
- Launching basic controller with mode switcher
  - Currently `<robot_host_name>` is `192.168.1.11`
```bash
python3 gesture/mode_manager.py <robot_host_name>
```
  - Initialisation of basic gestures for designated TAMP algorithm.
```bash
python3 Examples/gesturePR3.py <robot_host_name>
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
- **panda_py** (controller for Franka PR3)

Install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
