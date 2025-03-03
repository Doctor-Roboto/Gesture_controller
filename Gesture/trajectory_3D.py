#!/usr/bin/env python 

import cv2

class Trajectory3DMode:
    def __init__(self, sample_interval=5, z_scale=100):
        """
        Initializes the 3D trajectory mode.
        
        Args:
            sample_interval (int): Number of frames between sampling points.
            z_scale (float): Scale factor for the normalized z coordinate.
        """
        self.sample_interval = sample_interval
        self.frame_counter = 0
        # Each point is a tuple: (x_pixel, y_pixel, z_value)
        self.trajectory_points = []
        self.z_scale = z_scale

    def reset(self):
        """Clears stored trajectory data and resets the frame counter."""
        self.trajectory_points = []
        self.frame_counter = 0

    def update(self, hand_landmarks, frame_shape):
        """
        Updates the 3D trajectory based on the current hand landmarks.
        
        Args:
            hand_landmarks: The hand landmarks detected by MediaPipe.
            frame_shape: Tuple (height, width, channels) of the current frame.
        """
        self.frame_counter += 1
        if self.frame_counter % self.sample_interval == 0:
            wrist = hand_landmarks.landmark[0]
            h, w, _ = frame_shape
            x_pixel = int(w * wrist.x)
            y_pixel = int(h * wrist.y) 
            z_value = wrist.z * self.z_scale
            self.trajectory_points.append((x_pixel, y_pixel, z_value))
            print("3D Trajectory point:", (x_pixel, y_pixel, z_value))

    def draw(self, frame):
        """
        Draws the 3D trajectory onto the given frame.
        The trajectory is drawn using the 2D (x,y) projection.
        The line color is determined by the average z value between points:
          - Blue: moving into the screen.
          - Red: moving out of the screen.
          - Green: near the neutral plane.
        
        Args:
            frame: The current video frame (NumPy array).
        """
        if len(self.trajectory_points) < 2:
            return

        for i in range(1, len(self.trajectory_points)):
            pt1 = self.trajectory_points[i-1]
            pt2 = self.trajectory_points[i]
            # Compute average z to decide color.
            avg_z = (pt1[2] + pt2[2]) / 2
            if avg_z < -1:
                color = (255, 0, 0)  # Blue for "into" the screen.
            elif avg_z > 1:
                color = (0, 0, 255)  # Red for "out of" the screen.
            else:
                color = (0, 255, 0)  # Green for near neutral.
            cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color, 2)
