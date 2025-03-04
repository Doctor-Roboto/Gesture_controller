#!/usr/bin/env python 

"""
This is a similar launcher to the mode_manager.py script 
 - To launch: <python3 data_collection.py 192.168.1.11>
In this instance, data is recorded and stored as a list and plotted

------ Instructions on switcher -----
Numbers are in reference to holding up fingers: 2 is two fingers and 3 is three

Using gesture 2 will switch to gesture mode "okay" to confirm "stop" to deny
Using gesture 3 will switch to gesture mode "okay" to confirm "stop" to deny
While in trajectory mode can switch between  2D and 3D controls using 3

At the end of the session, ESC closes the  video window, and data will be displayed
**** Must save as windows appear or data will be lost ****

"""


import cv2
import mediapipe as mp
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

# Import my modules
from gesture.gestures import GestureRecognizer, OkayGesture, StopGesture
from gesture.trajectory import TrajectoryMode
from gesture.trajectory_3D import Trajectory3DMode

# Set this flag to False to disable data collection/plotting
DATA_COLLECTION = True

# Global containers for collected data.
collected_2d = []      # List of lists; each inner list is one 2D trajectory (list of (x,y) tuples)
collected_3d = []      # List of lists; each inner list is one 3D trajectory (list of (x,y,z) tuples)
collected_gestures = []  # List of tuples: (timestamp, gesture_name, command_value)

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
    
    # Threshold for finger postions
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

    # Instantiate modules.
    gesture_recognizer = GestureRecognizer(confirmation_frames=10)
    confirmation_okay = OkayGesture()
    confirmation_stop = StopGesture()

    # Mode variables
    # current_mode can be "gesture" or "trajectory".
    current_mode = "gesture"
    trajectory_submode = "2d"
    mode_switch_candidate = None
    mode_switch_counter = 0
    confirmation_required = False
    confirmation_counter_okay = 0
    confirmation_counter_stop = 0
    candidate_threshold = 10
    confirmation_threshold = 10

    # Change intervals here to increase frequency of data recording
    trajectory_mode_obj = TrajectoryMode(sample_interval=5)
    trajectory3d_mode_obj = Trajectory3DMode(sample_interval=5, z_scale=100)
    last_mode = current_mode
    last_trajectory_submode = trajectory_submode

    with mp_hands.Hands(min_detection_confidence=0.7,
                        min_tracking_confidence=0.5,
                        max_num_hands=1) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip and convert frame.
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
                    if current_mode != "trajectory":
                        if detect_number_gesture(hand_landmarks, 2) and current_mode != "gesture":
                            candidate_detected = "gesture"
                        elif detect_number_gesture(hand_landmarks, 3) and current_mode != "trajectory":
                            candidate_detected = "trajectory"
                    else:
                        # In trajectory mode, allow toggling between 2D and 3D using "3",
                        # and switching back to gesture mode with "2".
                        if detect_number_gesture(hand_landmarks, 3):
                            candidate_detected = "toggle_trajectory"
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

                    # Confirmation phase.
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
                            # Before switching modes, record any active trajectory session.
                            if current_mode == "trajectory":
                                if trajectory_submode == "2d" and len(trajectory_mode_obj.trajectory_points) > 0:
                                    collected_2d.append(list(trajectory_mode_obj.trajectory_points))
                                    trajectory_mode_obj.reset()
                                elif trajectory_submode == "3d" and len(trajectory3d_mode_obj.trajectory_points) > 0:
                                    collected_3d.append(list(trajectory3d_mode_obj.trajectory_points))
                                    trajectory3d_mode_obj.reset()

                            if mode_switch_candidate == "toggle_trajectory":
                                # Toggling between 2D and 3D within trajectory mode.
                                if trajectory_submode == "2d":
                                    trajectory_submode = "3d"
                                    print("Trajectory submode switched to 3D")
                                else:
                                    trajectory_submode = "2d"
                                    print("Trajectory submode switched to 2D")
                            else:
                                # Switching main modes.
                                current_mode = mode_switch_candidate
                                print("Mode switched to:", current_mode)
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

            # Record gesture commands in gesture mode.
            if current_mode == "gesture" and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    confirmed_gesture = gesture_recognizer.update(hand_landmarks)
                    if confirmed_gesture:
                        gesture_name, command_value = confirmed_gesture
                        cv2.putText(frame, f"{gesture_name} ({command_value})", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print(f"Gesture command: {gesture_name}, {command_value}")
                        # Record with a timestamp.
                        if DATA_COLLECTION:
                            collected_gestures.append((time.time(), gesture_name, command_value))

            # Mode-specific processing.
            if current_mode == "trajectory":
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    if trajectory_submode == "2d":
                        trajectory_mode_obj.update(hand_landmarks, frame.shape)
                        trajectory_mode_obj.draw(frame)
                    elif trajectory_submode == "3d":
                        trajectory3d_mode_obj.update(hand_landmarks, frame.shape)
                        trajectory3d_mode_obj.draw(frame)

            # Display current mode info.
            cv2.putText(frame, f"Current Mode: {current_mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if current_mode == "trajectory":
                cv2.putText(frame, f"Trajectory Submode: {trajectory_submode}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            if confirmation_required and mode_switch_candidate:
                cv2.putText(frame, f"Confirm switch to {mode_switch_candidate}? (OK/STOP)", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Mode Manager", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                # Before quitting, record any active session
                if current_mode == "trajectory":
                    if trajectory_submode == "2d" and len(trajectory_mode_obj.trajectory_points) > 0:
                        collected_2d.append(list(trajectory_mode_obj.trajectory_points))
                        trajectory_mode_obj.reset()
                    elif trajectory_submode == "3d" and len(trajectory3d_mode_obj.trajectory_points) > 0:
                        collected_3d.append(list(trajectory3d_mode_obj.trajectory_points))
                        trajectory3d_mode_obj.reset()
                break

    cap.release()
    cv2.destroyAllWindows()

    # --- Data Collection: Plotting & Summary ---
    if DATA_COLLECTION:
        # Plot each 2D trajectory in its own figure
        for i, traj in enumerate(collected_2d):
            if len(traj) < 2:
                continue
            x_vals = [pt[0] for pt in traj]
            y_vals = [pt[1] for pt in traj]
            plt.figure()
            plt.plot(x_vals, y_vals, marker='o', linestyle='-')
            plt.title(f"2D Trajectory {i+1}")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            plt.gca().invert_yaxis()  
            plt.grid(True)
        
        # Plot each 3D trajectory in its own figure
        for i, traj in enumerate(collected_3d):
            if len(traj) < 2:
                continue
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x_vals = [pt[0] for pt in traj]
            y_vals = [pt[1] for pt in traj]
            z_vals = [pt[2] for pt in traj]
            ax.plot(x_vals, y_vals, z_vals, marker='o')
            ax.set_title(f"3D Trajectory {i+1}")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.set_zlabel("Z (scaled)")
            plt.grid(True)
        
        # Plot gesture summary
        if collected_gestures:
            times = [t - collected_gestures[0][0] for (t, _, _) in collected_gestures]
            gesture_vals = [cmd for (_, _, cmd) in collected_gestures]
            gesture_names = [name for (_, name, _) in collected_gestures]
            plt.figure()
            plt.scatter(times, gesture_vals, c='m')
            for i, name in enumerate(gesture_names):
                plt.annotate(name, (times[i], gesture_vals[i]))
            plt.title("Gesture Summary")
            plt.xlabel("Time (s)")
            plt.ylabel("Command Value")
            plt.grid(True)

        plt.show()

if __name__ == "__main__":
    main()
