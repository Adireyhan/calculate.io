import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import collections

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.co:19302"]},
        ]
    }
)
mp_pose = mp.solutions.pose

# Gunakan VideoProcessorBase, bukan VideoTransformerBase
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.baseline = None
        self.y_history = collections.deque(maxlen=3)
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1
        )

    # Nama fungsi tetap recv (atau transform), tapi recv lebih standar sekarang
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) 
        
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

            if avg_y < self.baseline - jump_threshold and self.stage == "down":
                self.stage = "up"
            elif avg_y > self.baseline - (jump_threshold * 0.2) and self.stage == "up":
                self.counter += 1
                self.stage = "down"
                self.baseline = (self.baseline * 0.9) + (avg_y * 0.1)

            cv2.rectangle(img, (0, 0), (280, 120), (245, 117, 16), -1)
            cv2.putText(img, f"REPS: {self.counter}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(img, f"STAGE: {self.stage}", (10, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Harus mengembalikan VideoFrame
        import av
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="AI Skipping Counter", page_icon="🏃")
st.title("🏃 AI Skipping Counter")

# Pemanggilan webrtc_streamer yang diperbarui
webrtc_streamer(
    key="skipping-counter",
    # Gunakan video_processor_factory (BUKAN video_transformer_factory)
    video_processor_factory=PoseProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # Menambah performa
)
