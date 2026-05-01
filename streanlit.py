import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import collections

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose

class PoseTransformer(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.baseline = None
        self.y_history = collections.deque(maxlen=3)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip biar kayak cermin
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
            self.y_history.append(nose_y)
            avg_y = sum(self.y_history) / len(self.y_history)

            shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            body_size = abs(hip_y - shoulder_y)
            jump_threshold = body_size * 0.30

            if self.baseline is None:
                self.baseline = avg_y

            # Logika Hitung
            if avg_y < self.baseline - jump_threshold and self.stage == "down":
                self.stage = "up"
            elif avg_y > self.baseline - (jump_threshold * 0.2) and self.stage == "up":
                self.counter += 1
                self.stage = "down"
                self.baseline = (self.baseline * 0.9) + (avg_y * 0.1)

            # Gambar visualisasi di frame
            cv2.putText(img, f"Reps: {self.counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Stage: {self.stage}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

st.title("AI Skipping Counter")
webrtc_streamer(key="example", video_transformer_factory=PoseTransformer)