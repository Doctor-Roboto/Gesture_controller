#!/usr/bin/env python 

"""
First launch gesture controller contains directional controls (Up, Down, Left, Right)
As well as confimation controllers of `Stop` and `Okay`

To be launched using mode_manager.py or data_colelction.py

----- EDITS --------
To add gestures define Class and landmarks
"""


import math

class Gesture:
    def __init__(self, name, command_value):
        self.name = name
        self.command_value = command_value

    def detect(self, hand_landmarks):
        """
        Should return True if this gesture is detected.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class StopGesture(Gesture):
    def __init__(self, threshold=0.3):
        super().__init__("stop", 0)
        self.threshold = threshold

    def detect(self, hand_landmarks):
        wrist      = hand_landmarks.landmark[0]
        index_tip  = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip   = hand_landmarks.landmark[16]
        pinky_tip  = hand_landmarks.landmark[20]

        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        if (dist(index_tip, wrist) > self.threshold and
            dist(middle_tip, wrist) > self.threshold and
            dist(ring_tip, wrist) > self.threshold and
            dist(pinky_tip, wrist) > self.threshold):
            return True
        return False


class OkayGesture(Gesture):
    def __init__(self, index_thumb_threshold=0.05, finger_threshold=0.2):
        super().__init__("okay", 1)
        self.index_thumb_threshold = index_thumb_threshold
        self.finger_threshold = finger_threshold

    def detect(self, hand_landmarks):
        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        thumb_tip  = hand_landmarks.landmark[4]
        index_tip  = hand_landmarks.landmark[8]
        wrist      = hand_landmarks.landmark[0]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip   = hand_landmarks.landmark[16]
        pinky_tip  = hand_landmarks.landmark[20]

        if dist(thumb_tip, index_tip) < self.index_thumb_threshold:
            if (dist(middle_tip, wrist) > self.finger_threshold and
                dist(ring_tip, wrist) > self.finger_threshold and
                dist(pinky_tip, wrist) > self.finger_threshold):
                return True
        return False


class GoUpGesture(Gesture):
    def __init__(self, finger_curled_threshold=0.25):
        super().__init__("go up", 2)
        self.finger_curled_threshold = finger_curled_threshold

    def detect(self, hand_landmarks):
        thumb_tip  = hand_landmarks.landmark[4]
        thumb_ip   = hand_landmarks.landmark[3]
        wrist      = hand_landmarks.landmark[0]
        index_tip  = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip   = hand_landmarks.landmark[16]
        pinky_tip  = hand_landmarks.landmark[20]

        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        # "Thumbs up": thumb tip above its IP joint and other fingers curled.
        if thumb_tip.y < thumb_ip.y:
            if (dist(index_tip, wrist) < self.finger_curled_threshold and
                dist(middle_tip, wrist) < self.finger_curled_threshold and
                dist(ring_tip, wrist) < self.finger_curled_threshold and
                dist(pinky_tip, wrist) < self.finger_curled_threshold):
                return True
        return False


class GoDownGesture(Gesture):
    def __init__(self, finger_curled_threshold=0.25):
        super().__init__("go down", 3)
        self.finger_curled_threshold = finger_curled_threshold

    def detect(self, hand_landmarks):
        thumb_tip  = hand_landmarks.landmark[4]
        thumb_ip   = hand_landmarks.landmark[3]
        wrist      = hand_landmarks.landmark[0]
        index_tip  = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip   = hand_landmarks.landmark[16]
        pinky_tip  = hand_landmarks.landmark[20]

        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        # "Thumbs down": thumb tip below its IP joint and other fingers curled.
        if thumb_tip.y > thumb_ip.y:
            if (dist(index_tip, wrist) < self.finger_curled_threshold and
                dist(middle_tip, wrist) < self.finger_curled_threshold and
                dist(ring_tip, wrist) < self.finger_curled_threshold and
                dist(pinky_tip, wrist) < self.finger_curled_threshold):
                return True
        return False


class PointingSpatialGesture(Gesture):
    """
    Detects a pointing gesture (index finger extended while the others are folded)
    and then uses the index finger tip's normalized coordinates to assign a spatial
    command based on four screen areas (with a dead zone near the center).
    """
    def __init__(self, min_index_dist=0.4, max_other_finger_dist=0.3,
                 dead_zone_min=0.4, dead_zone_max=0.6):
        # Initial name and command_value are placeholders; they will be updated dynamically.
        super().__init__("pointing spatial", None)
        self.min_index_dist = min_index_dist
        self.max_other_finger_dist = max_other_finger_dist
        self.dead_zone_min = dead_zone_min
        self.dead_zone_max = dead_zone_max

    def detect(self, hand_landmarks):
        wrist      = hand_landmarks.landmark[0]
        index_tip  = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip   = hand_landmarks.landmark[16]
        pinky_tip  = hand_landmarks.landmark[20]

        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        # Ensure the index finger is sufficiently extended.
        if dist(wrist, index_tip) < self.min_index_dist:
            return False

        # Ensure the other fingers are folded.
        if (dist(wrist, middle_tip) > self.max_other_finger_dist or
            dist(wrist, ring_tip)   > self.max_other_finger_dist or
            dist(wrist, pinky_tip)  > self.max_other_finger_dist):
            return False

        x = index_tip.x
        y = index_tip.y

        # Dead zone: if the index tip is near the center, ignore the spatial command.
        if self.dead_zone_min < x < self.dead_zone_max and self.dead_zone_min < y < self.dead_zone_max:
            return False

        # Determine the quadrant relative to the center (0.5, 0.5).
        direction = ""
        if y < 0.5:
            direction += "top "
        elif y >= 0.5:
            direction += "bottom "

        if x < 0.5:
            direction += "left"
        elif x >= 0.5:
            direction += "right"

        # Update the gesture name and assign a command value
        self.name = f"go {direction.strip()}"
        mapping = {
            "go top left": 5,
            "go top right": 6,
            "go bottom left": 7,
            "go bottom right": 8
        }
        self.command_value = mapping.get(self.name, None)
        return True


# class ComeHereGesture(Gesture):
#     def __init__(self, horizontal_threshold=0.2):
#         super().__init__("come here", 4)
#         self.horizontal_threshold = horizontal_threshold

#     def detect(self, hand_landmarks):
#         wrist     = hand_landmarks.landmark[0]
#         index_tip = hand_landmarks.landmark[8]
#         if abs(index_tip.x - wrist.x) > self.horizontal_threshold:
#             return True
#         return False


class GestureRecognizer:
    """
    Aggregates available gesture detectors and adds a confirmation mechanism:
    a gesture must be detected for a defined number of consecutive frames
    before it is confirmed.
    """
    def __init__(self, confirmation_frames=10):
        self.gestures = [
            OkayGesture(),
            StopGesture(),
            GoUpGesture(),
            GoDownGesture(),
            PointingSpatialGesture(), 
            # ComeHereGesture()
        ]
        self.confirmation_frames = confirmation_frames
        self.last_gesture = None 
        self.frame_count = 0

    def update(self, hand_landmarks):
        """
        Processes the current frame's hand landmarks.
        Returns a tuple (gesture_name, command_value) if a gesture is confirmed
        """
        detected_gesture = None
        for gesture in self.gestures:
            if gesture.detect(hand_landmarks):
                detected_gesture = gesture
                break  # Stop after the first matching gesture

        if detected_gesture and self.last_gesture and detected_gesture.name == self.last_gesture.name:
            self.frame_count += 1
        else:
            self.last_gesture = detected_gesture
            self.frame_count = 1 if detected_gesture else 0

        if self.frame_count >= self.confirmation_frames and detected_gesture:
            confirmed = (detected_gesture.name, detected_gesture.command_value)
            self.last_gesture = None
            self.frame_count = 0
            return confirmed

        return None
