#!/usr/bin/env python 

"""
This is a feedback loop based around obstacle placement while the robot
is moving to a designated goal
You can use either text or gesture as input to adjust trajectory while
in motion, although there is a spatial "deadzone" to prevent collisions 

-------- This is specifically designed for Franka PR3 ----------
The postional arguements will be changed using either a sperate script or
manual adjustments to destination or obstacle placement. 

To launch -- python3 spatial_controller.py 192.168.1.11

"""

import numpy as np
import panda_py
from panda_py import controllers
import threading
from queue import Queue
import time
import sys
import cv2
import mediapipe as mp

# Import the built gesture library.
from gesture.gestures import (GestureRecognizer, OkayGesture, StopGesture,
    GoUpGesture, GoDownGesture, PointingSpatialGesture)

def bezier_curve(points, num_points=100):
    """
    Generates a quadratic Bezier curve given three 3D points: start, control, and target.
    
    Args:
        points: List of three 3D points.
        num_points: Number of discrete points along the curve.
    
    Returns:
        NumPy array of shape (num_points, 3) with the trajectory.
    """
    if len(points) != 3:
        raise ValueError("Expected 3 control points for a quadratic Bezier curve.")
    p0, p1, p2 = points
    t_values = np.linspace(0, 1, num_points)
    curve = np.array([
        (1 - t)**2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t**2 * np.array(p2)
        for t in t_values
    ])
    return curve

class RobotController:
    def __init__(self, panda, obstacles):
        """
        Initializes the robot controller.
        
        Args:
            panda: The robot interface.
            obstacles: A list of obstacles. Each obstacle is a dict with keys 'min' and 'max',
                       representing the minimum and maximum 3D coordinates of the obstacle.
        """
        self.panda = panda
        self.curve_points = None
        self.control_lock = threading.Lock()
        self.update_queue = Queue()
        self.running = True
        self.obstacles = obstacles
        self.paused = False
        # Instantiate the gesture recognizer from your built gesture library.
        self.gesture_recognizer = GestureRecognizer(confirmation_frames=10)

    def get_control_point_offset(self, modifier):
        """
        Returns an offset vector based on the modifier command.
        Supports composite commands (e.g., "up_left") by summing basic offsets.
        """
        offset = np.array([0.0, 0.0, 0.0])
        if "right" in modifier:
            offset += np.array([0.0, 0.2, 0.0])
        if "left" in modifier:
            offset += np.array([0.0, -0.2, 0.0])
        if "up" in modifier:
            offset += np.array([0.0, 0.0, 0.2])
        if "down" in modifier:
            offset += np.array([0.0, 0.0, -0.2])
        return offset

    def calculate_control_point(self, start, target, modifier):
        """
        Calculates a control point between start and target based on the modifier.
        """
        midpoint = (np.array(start) + np.array(target)) / 2.0
        offset = self.get_control_point_offset(modifier)
        return (midpoint + offset).tolist()

    def adjust_for_obstacles(self, position, margin=0.05):
        """
        Gradually adjusts the given position if it is within a safety margin of any obstacle.
        
        For each obstacle, an expanded region is defined by subtracting/adding the margin.
        If the position falls inside this expanded region (but outside the obstacle itself),
        a small offset (here, upward) is added to help steer the robot away gradually.
        
        Args:
            position: The original 3D position (list or NumPy array).
            margin: The safety margin distance.
            
        Returns:
            Adjusted 3D position as a NumPy array.
        """
        pos = np.array(position)
        adjustment = np.zeros(3)
        for obs in self.obstacles:
            expanded_min = obs['min'] - margin
            expanded_max = obs['max'] + margin
            if np.all(pos >= expanded_min) and np.all(pos <= expanded_max):
                delta = (expanded_min[2] + margin) - pos[2]
                if delta > 0:
                    adjustment[2] += delta
                else:
                    delta_top = pos[2] - (expanded_max[2] - margin)
                    if delta_top > 0:
                        adjustment[2] -= delta_top
        return pos + adjustment

    def map_gesture_to_modifier(self, gesture_name):
        """
        Maps a recognized gesture name from the built gesture library to a modifier command.
        Recognized gestures should include:
          - "stop" -> "stop"
          - "okay" -> "okay"
          - "go up" -> "up"
          - "go down" -> "down"
          - "go top left" -> "up_left"
          - "go top right" -> "up_right"
          - "go bottom left" -> "down_left"
          - "go bottom right" -> "down_right"
          - Otherwise, defaults to "direct"
        """
        g = gesture_name.lower()
        if "stop" in g:
            return "stop"
        elif "okay" in g:
            return "okay"
        elif "go up" in g:
            if "left" in g:
                return "up_left"
            elif "right" in g:
                return "up_right"
            return "up"
        elif "go down" in g:
            if "left" in g:
                return "down_left"
            elif "right" in g:
                return "down_right"
            return "down"
        elif "left" in g:
            return "left"
        elif "right" in g:
            return "right"
        else:
            return "direct"

    def move_to_target(self, target, ctrl, modifier="direct"):
        """
        Generates a trajectory to the target and executes the control loop.
        Dynamically updates the path if new textual or gestural commands are received.
        Additionally, adjusts each desired position gradually when near an obstacle.
        """
        # Get starting position and generate initial trajectory.
        start = self.panda.get_position()
        control_point = self.calculate_control_point(start, target, modifier)
        points = [start, control_point, target]
        with self.control_lock:
            self.curve_points = bezier_curve(points, num_points=100)
        q0 = self.panda.get_orientation()
        runtime = 30.0  # seconds; adjust as needed
        self.panda.start_controller(ctrl)
        index = 0
        self.paused = False

        with self.panda.create_context(frequency=10, max_runtime=runtime) as ctx:
            while ctx.ok() and self.running:
                # Check for new modifier commands from text or gesture.
                if not self.update_queue.empty():
                    new_modifier = self.update_queue.get()
                    if new_modifier == "stop":
                        self.paused = True
                        print("Forward motion paused.")
                    elif new_modifier == "okay":
                        self.paused = False
                        print("Forward motion resumed.")
                    elif new_modifier in [
                        "right", "left", "up", "down",
                        "up_left", "up_right", "down_left", "down_right",
                        "direct"
                    ]:
                        current_pos = self.panda.get_position()
                        new_control_point = self.calculate_control_point(current_pos, target, new_modifier)
                        new_points = [current_pos, new_control_point, target]
                        with self.control_lock:
                            self.curve_points = bezier_curve(new_points, num_points=100)
                        index = 0
                        print(f"Updated trajectory with modifier: {new_modifier}")

                if self.paused:
                    time.sleep(0.1)
                    continue

                with self.control_lock:
                    if index < len(self.curve_points):
                        desired_position = self.curve_points[index]
                        index += 1
                    else:
                        break

                adjusted_position = self.adjust_for_obstacles(desired_position, margin=0.05)
                ctrl.set_control(adjusted_position, q0)
                time.sleep(0.1)

    def input_thread(self):
        """
        Thread to accept textual commands simulating LLM input.
        Valid commands: right, left, up, down, direct, stop, okay or quit.
        """
        while self.running:
            cmd = input("Enter modifier (right, left, up, down, direct, stop, okay) or 'quit': ").strip().lower()
            if cmd == "quit":
                self.running = False
                break
            if cmd in ["right", "left", "up", "down", "direct", "stop", "okay"]:
                self.update_queue.put(cmd)
            else:
                print("Invalid command.")

    def gesture_thread(self):
        """
        Thread to capture gestural commands using MediaPipe.
        Uses the built gesture library's GestureRecognizer to detect gestures,
        then maps the recognized gesture to a modifier command and places it in the update queue.
        Valid outputs: right, left, up, down, up_left, up_right, down_left, down_right, stop, okay, or direct.
        """
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(min_detection_confidence=0.7,
                            min_tracking_confidence=0.5,
                            max_num_hands=1) as hands:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        gesture = self.gesture_recognizer.update(hand_landmarks)
                        if gesture:
                            gesture_name, _ = gesture
                            modifier = self.map_gesture_to_modifier(gesture_name)
                            if modifier:
                                self.update_queue.put(modifier)
                                print(f"Gestural input: {modifier}")
                cv2.waitKey(5)
        cap.release()

def create_impedance_matrix(x_stiffness=300, y_stiffness=300, z_stiffness=300,
                              rx_stiffness=10, ry_stiffness=10, rz_stiffness=10):
    """
    Creates a diagonal impedance matrix for the Cartesian impedance controller.
    """
    return np.diag([x_stiffness, y_stiffness, z_stiffness, rx_stiffness, ry_stiffness, rz_stiffness])

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 spatial_feedback.py 192.168.1.11")
        sys.exit(1)

    robot_hostname = sys.argv[1]
    panda = panda_py.Panda(robot_hostname)
    panda.move_to_start()
    impedance_matrix = create_impedance_matrix()
    ctrl = controllers.CartesianImpedance(
        impedance=impedance_matrix,
        damping_ratio=1.0,
        filter_coeff=1.0
    )

    # Define obstacles, change dependant on postions
    obstacles = [
        {"min": np.array([0.5, 0.5, 0.2]), "max": np.array([0.7, 0.7, 0.4])},
        {"min": np.array([0.2, 0.3, 0.1]), "max": np.array([0.3, 0.4, 0.5])}
    ]

    target_position = [0.7, 0.2, 0.3]
    controller = RobotController(panda, obstacles)

    # Start textual input thread
    input_thread = threading.Thread(target=controller.input_thread)
    input_thread.daemon = True
    input_thread.start()

    # Start gestural input thread
    gesture_thread = threading.Thread(target=controller.gesture_thread)
    gesture_thread.daemon = True
    gesture_thread.start()

    controller.move_to_target(target_position, ctrl, modifier="direct")

    print("Motion complete or controller stopped.")
    controller.running = False
    input_thread.join()
    gesture_thread.join()

if __name__ == '__main__':
    main()
