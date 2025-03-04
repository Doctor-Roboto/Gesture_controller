#!/usr/bin/env python 

"""
This is specifically designed to work with Franka PR3 and current lab set up
Designed as an example of simple gestural commands to motion parameters

"""


import numpy as np
import panda_py
from panda_py import controllers
import sys
import threading
from queue import Queue
import time
import cv2
import mediapipe as mp

# Modules in this package
from Gesture_controller.Gesture.gestures import GestureRecognizer, StopGesture, OkayGesture

class RobotController:
    def __init__(self):
        self.curve_points = None
        self.running = True
        self.control_lock = threading.Lock()
        self.update_queue = Queue()
        self.paused = False  # When True, the robot stops advancing along its path.
        self.gesture_recognizer = GestureRecognizer(confirmation_frames=5)

    def bezier_curve_approach(self, points, num_points=50):
        """Generates a quadratic 3D Bezier curve through 3 control points."""
        if len(points) != 3:
            raise ValueError("Must provide exactly 3 control points.")
        p0, p1, p2 = points
        t_values = np.linspace(0, 1, num_points)
        bezier_points = np.array([
            [(1 - t) ** 2 * p0[i] + 2 * (1 - t) * t * p1[i] + t ** 2 * p2[i] for i in range(3)]
            for t in t_values
        ])
        return bezier_points

    def get_control_point_offset(self, modifier):
        """
        Returns an offset vector based on the modifier.
        'right': +0.3 along y; 'up': +0.3 along z; 'left': -0.3 along y;
        'down': -0.3 along z; 'direct': no offset.
        """
        if modifier == "right":
            return np.array([0.0, 0.3, 0.0])
        elif modifier == "up":
            return np.array([0.0, 0.0, 0.3])
        elif modifier == "left":
            return np.array([0.0, -0.3, 0.0])
        elif modifier == "down":
            return np.array([0.0, 0.0, -0.3])
        elif modifier == "direct":
            return np.array([0.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 0.0])

    def calculate_control_point(self, start_pt, end_pt, modifier):
        """Calculates a control point by taking the midpoint of start and end, then adding an offset."""
        midpoint = (np.array(start_pt) + np.array(end_pt)) / 2.0
        offset = self.get_control_point_offset(modifier)
        control_point = midpoint + offset
        return control_point.tolist()

    def move_approach_bezier(self, panda, end_point, ctrl, initial_modifier):
        """
        Commands the robot to follow a Bezier curve from its current position to end_point.
        If new directional commands are received via the update queue, the curve is recalculated.
        If paused (via a stop gesture), movement is held until resumed.
        """
        start_pt = panda.get_position()
        control_point = self.calculate_control_point(start_pt, end_point, initial_modifier)
        points = np.array([start_pt, control_point, end_point])
        with self.control_lock:
            self.curve_points = self.bezier_curve_approach(points)
        q0 = panda.get_orientation()
        runtime = 20.0
        panda.start_controller(ctrl)
        inc_counter = 0

        with panda.create_context(frequency=10, max_runtime=runtime) as ctx:
            while ctx.ok() and self.running:
                # If paused, hold the current position
                if self.paused:
                    print("Paused: Holding current position.")
                    time.sleep(0.1)
                    continue

                # Check for updated directional commands.
                if not self.update_queue.empty():
                    new_modifier = self.update_queue.get()
                    print("New directional command received:", new_modifier)
                    current_pos = panda.get_position()
                    new_control_point = self.calculate_control_point(current_pos, end_point, new_modifier)
                    new_points = np.array([current_pos, new_control_point, end_point])
                    with self.control_lock:
                        self.curve_points = self.bezier_curve_approach(new_points)
                    inc_counter = 0  # Restart along the new path

                # Move along the current Bezier path.
                with self.control_lock:
                    if inc_counter < len(self.curve_points):
                        x_d = self.curve_points[inc_counter]
                        inc_counter += 1
                        print(f"Moving to position: {x_d}")
                        ctrl.set_control(x_d, q0)
                    else:
                        break
                time.sleep(0.1)

    def map_gesture_to_modifier(self, gesture_name):
        """
        Maps a recognized gesture name to one of the directional modifiers.
        For example, if the gesture name contains 'up' or 'top', returns "up", etc.
        """
        lower = gesture_name.lower()
        if "up" in lower or "top" in lower:
            return "up"
        if "down" in lower or "bottom" in lower:
            return "down"
        if "left" in lower:
            return "left"
        if "right" in lower:
            return "right"
        return "direct"

    def gesture_thread_function(self):
        """
        Runs in a separate thread.
        Uses MediaPipe to capture hand landmarks and then the GestureRecognizer to detect gestures.
        Processes directional gestures by putting modifier strings into update_queue.
        A "stop" gesture sets the paused flag and an "okay" gesture resumes movement.
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
                        # Update our gesture recognizer with the current hand landmarks.
                        gesture = self.gesture_recognizer.update(hand_landmarks)
                        if gesture:
                            gesture_name, command_value = gesture
                            print("Gesture detected:", gesture_name, command_value)
                            # Check for stop/okay gestures.
                            if "stop" in gesture_name.lower():
                                self.paused = True
                                print("STOP gesture detected: Pausing movement.")
                            elif "okay" in gesture_name.lower():
                                self.paused = False
                                print("OKAY gesture detected: Resuming movement.")
                            else:
                                # Assume any other gesture is a directional command.
                                modifier = self.map_gesture_to_modifier(gesture_name)
                                if modifier:
                                    print("Mapping gesture to modifier:", modifier)
                                    self.update_queue.put(modifier)
                cv2.waitKey(5)
        cap.release()

def create_impedance_matrix(x_stiffness=300, y_stiffness=300, z_stiffness=300,
                            rx_stiffness=10, ry_stiffness=10, rz_stiffness=10):
    return np.diag([x_stiffness, y_stiffness, z_stiffness, rx_stiffness, ry_stiffness, rz_stiffness])

def move_to_joint(panda, joint_pos):
    start = np.array(joint_pos)
    panda.move_to_joint_position(start)

def main():
    if len(sys.argv) < 2:
        raise RuntimeError(f"Usage: python {sys.argv[0]} <robot-hostname>")
    
    # Initialize robot.
    panda = panda_py.Panda(sys.argv[1])
    robot_controller = RobotController()
    
    # Move to an initial joint position.
    start_joint = [
        0.6310074464236987, 0.16027098904586753, 0.9026972943092949, 
        -2.7143195967015252, -1.639964292837852, 1.501099148008071, 0.49099073080729655
    ]
    move_to_joint(panda, start_joint)
    
    # Create an impedance matrix and controller.
    impedance_matrix = create_impedance_matrix(x_stiffness=300, y_stiffness=300, z_stiffness=300)
    ctrl = controllers.CartesianImpedance(
        impedance=impedance_matrix,
        damping_ratio=1.0,
        filter_coeff=1.0
    )
    
    # Define the target position.
    upper_box_drawing = [0.7200819652082762, 0.249039807354959, 0.20541295537790964]
    
    # Start the gesture recognition thread.
    gesture_thread = threading.Thread(target=robot_controller.gesture_thread_function)
    gesture_thread.daemon = True
    gesture_thread.start()
    
    try:
        # Begin the movement toward the target using an initial "direct" modifier.
        robot_controller.move_approach_bezier(panda, upper_box_drawing, ctrl, initial_modifier="direct")
    finally:
        robot_controller.running = False
        gesture_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
