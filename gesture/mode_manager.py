#!/usr/bin/env python 

"""
Designed for integration of gestural controller for Franka PR3
Will initially launch with gesture mode for basic controls

------ Instuctions on switcher -----
Numers are in refence to holding up fingers 2 is two fingers and 3 is three

Using the gesture 2 will swith to gesture mode "okay" to confirm "stop" to deny
Using the gesture 3 will swith to gesture mode "okay" to confirm "stop" to deny
While in trajectory mode can swithc between  2D and 3D controls using 3


To launch <python3 mode_manager.py 192.168.1.11>
"""


import cv2
import mediapipe as mp
import math
from Gesture_controller.Gesture.gestures import GestureRecognizer, OkayGesture, StopGesture
from Gesture_controller.Gesture.trajectory import TrajectoryMode
from Gesture_controller.Gesture.trajectory_3D import Trajectory3DMode

def detect_number_gesture(hand_landmarks, number):
    """
    Detects a simple number gesture.
    
    For number '2': index and middle fingers extended; ring and pinky folded.
    For number '3': index, middle, and ring fingers extended; pinky folded.
    """
    def dist(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    
    wrist = hand_landmarks.landmark[0]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Threshold for determining extended vs. folded.
    extended_threshold = 0.3
    index_extended = dist(wrist, index_tip) > extended_threshold
    middle_extended = dist(wrist, middle_tip) > extended_threshold
    ring_extended = dist(wrist, ring_tip) > extended_threshold
    pinky_extended = dist(wrist, pinky_tip) > extended_threshold
    
    if number == 2:
        if index_extended and middle_extended and (not ring_extended) and (not pinky_extended):
            return True
    elif number == 3:
        if index_extended and middle_extended and ring_extended and (not pinky_extended):
            return True
    return False

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    # Instantiate gesture recognizer for gesture mode.
    gesture_recognizer = GestureRecognizer(confirmation_frames=10)
    # Instantiate confirmation detectors.
    confirmation_okay = OkayGesture()
    confirmation_stop = StopGesture()

    # Mode variables.
    # current_mode can be "gesture" or "trajectory".
    current_mode = "gesture"
    # When in trajectory mode, trajectory_submode can be "2d" or "3d".
    trajectory_submode = "2d"
    mode_switch_candidate = None
    mode_switch_counter = 0
    confirmation_required = False
    confirmation_counter_okay = 0
    confirmation_counter_stop = 0
    candidate_threshold = 10
    confirmation_threshold = 10

    # Instantiate trajectory mode objects.
    trajectory_mode_obj = TrajectoryMode(sample_interval=5)
    trajectory3d_mode_obj = Trajectory3DMode(sample_interval=5, z_scale=100)

    with mp_hands.Hands(min_detection_confidence=0.7,
                        min_tracking_confidence=0.5,
                        max_num_hands=1) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip and convert the frame.
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            results = hands.process(frame_rgb)

            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            candidate_detected = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # --- Mode Switching Logic ---
                    # If NOT in trajectory mode, use number gestures to switch modes.
                    if current_mode != "trajectory":
                        if detect_number_gesture(hand_landmarks, 2) and current_mode != "gesture":
                            candidate_detected = "gesture"
                        elif detect_number_gesture(hand_landmarks, 3) and current_mode != "trajectory":
                            candidate_detected = "trajectory"
                    else:
                        # If already in trajectory mode, allow sub-mode toggling:
                        # "3" toggles between 2D and 3D.
                        if detect_number_gesture(hand_landmarks, 3):
                            candidate_detected = "toggle_trajectory"
                        # Also allow switching back to gesture mode with "2".
                        elif detect_number_gesture(hand_landmarks, 2):
                            candidate_detected = "gesture"

                    # Update candidate counters.
                    if candidate_detected:
                        if mode_switch_candidate == candidate_detected:
                            mode_switch_counter += 1
                        else:
                            mode_switch_candidate = candidate_detected
                            mode_switch_counter = 1
                        if mode_switch_counter >= candidate_threshold:
                            confirmation_required = True
                            confirmation_counter_okay = 0
                            confirmation_counter_stop = 0
                    else:
                        mode_switch_counter = max(0, mode_switch_counter - 1)

                    # Confirmation phase using the "okay" (confirm) and "stop" (cancel) gestures.
                    if confirmation_required:
                        if confirmation_okay.detect(hand_landmarks):
                            confirmation_counter_okay += 1
                        else:
                            confirmation_counter_okay = max(0, confirmation_counter_okay - 1)
                        if confirmation_stop.detect(hand_landmarks):
                            confirmation_counter_stop += 1
                        else:
                            confirmation_counter_stop = max(0, confirmation_counter_stop - 1)

                        if confirmation_counter_okay >= confirmation_threshold:
                            # Confirm the candidate.
                            if mode_switch_candidate == "toggle_trajectory":
                                # Toggle between 2D and 3D sub-modes.
                                if trajectory_submode == "2d":
                                    trajectory_submode = "3d"
                                    print("Trajectory submode switched to 3D")
                                else:
                                    trajectory_submode = "2d"
                                    print("Trajectory submode switched to 2D")
                                # Reset trajectory data.
                                trajectory_mode_obj.reset()
                                trajectory3d_mode_obj.reset()
                            else:
                                # For switching between main modes.
                                current_mode = mode_switch_candidate
                                print("Mode switched to:", current_mode)
                                if current_mode == "trajectory":
                                    # Default to 2D when entering trajectory mode.
                                    trajectory_submode = "2d"
                                    trajectory_mode_obj.reset()
                                    trajectory3d_mode_obj.reset()
                            mode_switch_candidate = None
                            mode_switch_counter = 0
                            confirmation_required = False
                            confirmation_counter_okay = 0
                            confirmation_counter_stop = 0
                            break  
                        elif confirmation_counter_stop >= confirmation_threshold:
                            print("Mode switch canceled.")
                            mode_switch_candidate = None
                            mode_switch_counter = 0
                            confirmation_required = False
                            confirmation_counter_okay = 0
                            confirmation_counter_stop = 0

            # Display the current mode and (if in trajectory) the sub-mode.
            cv2.putText(frame, f"Current Mode: {current_mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if current_mode == "trajectory":
                cv2.putText(frame, f"Trajectory Submode: {trajectory_submode}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            if confirmation_required and mode_switch_candidate:
                cv2.putText(frame, f"Confirm switch to {mode_switch_candidate}? (OK/STOP)", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- Mode-Specific Processing ---
            if current_mode == "gesture":
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        confirmed_gesture = gesture_recognizer.update(hand_landmarks)
                        if confirmed_gesture:
                            gesture_name, command_value = confirmed_gesture
                            cv2.putText(frame, f"{gesture_name} ({command_value})", (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            print(f"Gesture command: {gesture_name}, {command_value}")
            elif current_mode == "trajectory":
                if results.multi_hand_landmarks:
                    # Use the first detected hand.
                    hand_landmarks = results.multi_hand_landmarks[0]
                    if trajectory_submode == "2d":
                        trajectory_mode_obj.update(hand_landmarks, frame.shape)
                        trajectory_mode_obj.draw(frame)
                    elif trajectory_submode == "3d":
                        trajectory3d_mode_obj.update(hand_landmarks, frame.shape)
                        trajectory3d_mode_obj.draw(frame)

            cv2.imshow("Mode Manager", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
