#!/usr/bin/env python 

import cv2

class TrajectoryMode:
    def __init__(self, sample_interval=5):
        """
        Initializes the trajectory mode.
        
        Args:
            sample_interval (int): Number of frames between sampling points.
        """
        self.sample_interval = sample_interval
        self.frame_counter = 0
        self.trajectory_points = []

    def reset(self):
        """
        Clears stored trajectory data and resets the frame counter.
        """
        self.trajectory_points = []
        self.frame_counter = 0

    def update(self, hand_landmarks, frame_shape):
        """
        Updates the trajectory based on the current hand landmarks.
        
        Args:
            hand_landmarks: The hand landmarks detected by MediaPipe.
            frame_shape: A tuple (height, width, channels) for converting 
                         normalized coordinates to pixels.
        """
        self.frame_counter += 1
        if self.frame_counter % self.sample_interval == 0:
            wrist = hand_landmarks.landmark[0]
            h, w, _ = frame_shape
            x_pixel = int(w * wrist.x)
            y_pixel = int(h * wrist.y)
            self.trajectory_points.append((x_pixel, y_pixel))
            print("Trajectory point:", (x_pixel, y_pixel))

    def draw(self, frame):
        """
        Draws the trajectory as a polyline on the given frame.
        
        Args:
            frame: The current video frame (as a NumPy array).
        """
        if len(self.trajectory_points) >= 2:
            for i in range(1, len(self.trajectory_points)):
                cv2.line(frame, self.trajectory_points[i - 1], self.trajectory_points[i], (0, 255, 255), 2)
