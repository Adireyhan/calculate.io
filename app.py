import streamlit as st
import mediapipe as mp
import cv2

# Inisialisasi
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseTransformer(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.baseline = None
        self.y_history = collections.deque(maxlen=3)
        # Inisialisasi model di dalam __init__ agar lebih efisien
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Efek cermin agar user nyaman
        
        # Proses gambar
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Ambil koordinat hidung
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
            self.y_history.append(nose_y)
            avg_y = sum(self.y_history) / len(self.y_history)

            # Hitung skala tubuh (Bahu ke Pinggul) untuk ambang batas dinamis
            shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            body_size = abs(hip_y - shoulder_y)
            jump_threshold = body_size * 0.30

            if self.baseline is None:
                self.baseline = avg_y

            # Logika Hitung Repetisi
            if avg_y < self.baseline - jump_threshold and self.stage == "down":
                self.stage = "up"
            elif avg_y > self.baseline - (jump_threshold * 0.2) and self.stage == "up":
                self.counter += 1
                self.stage = "down"
                # Update baseline dikit-dikit biar ngikutin posisi berdiri user
                self.baseline = (self.baseline * 0.9) + (avg_y * 0.1)

            # Overlay Text ke Video
            cv2.rectangle(img, (0, 0), (280, 120), (245, 117, 16), -1)
            cv2.putText(img, f"REPS: {self.counter}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(img, f"STAGE: {self.stage}", (10, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img

# Tampilan Web Streamlit
st.set_page_config(page_title="AI Skipping Counter", page_icon="🏃")
st.title("🏃 AI Skipping Counter")
st.write("Pastikan seluruh badan terlihat di kamera. Lompat untuk mulai menghitung!")

# Jalankan Streamer
webrtc_streamer(
    key="skipping-counter",
    mode="transform",
    video_transformer_factory=PoseTransformer,
    rtc_configuration=RTC_CONFIGURATION, # Penting buat koneksi HP
    media_stream_constraints={"video": True, "audio": False}, # Matikan audio biar hemat bandwidth
)

st.info("Tips: Jika baseline berantakan, refresh halaman untuk reset hitungan.")
