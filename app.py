import streamlit as st
import cv2
import time
import subprocess
import numpy as np
from ultralytics import YOLO

st.set_page_config(
    page_title="Sonic Flashlight",
    page_icon="🔦",
    layout="centered"
)

st.title("Sonic Flashlight")
st.caption("AI-powered assistive navigation device for the visually impaired")
st.divider()

st.sidebar.title("Settings")

source = st.sidebar.radio(
    "Camera Source",
    ["MacBook Webcam", "ESP32-CAM Stream"]
)

esp32_url = ""
if source == "ESP32-CAM Stream":
    esp32_url = st.sidebar.text_input(
        "ESP32 Stream URL",
        placeholder="http://192.168.1.45:81/stream"
    )

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.15,
    step=0.05
)

speak_results = st.sidebar.checkbox("Speak results aloud", value=True)
st.sidebar.divider()
st.sidebar.caption("Sonic Flashlight v1.0")
st.sidebar.caption("YOLO26n + MPS Acceleration")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "last_detected" not in st.session_state:
    st.session_state.last_detected = []
if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

with st.spinner("Loading YOLO26n model..."):
    model = load_model()

st.success("Model loaded — ready to scan")

def get_position(box, frame_width):
    x1, y1, x2, y2 = box
    centre_x = (x1 + x2) / 2
    ratio = centre_x / frame_width
    if ratio < 0.33:
        return "on your left"
    elif ratio > 0.66:
        return "on your right"
    else:
        return "in front of you"

col1, col2 = st.columns(2)
with col1:
    scan_button = st.button("Scan Now", use_container_width=True, type="primary")
with col2:
    clear_button = st.button("Clear History", use_container_width=True)

if clear_button:
    st.session_state.history = []
    st.session_state.last_frame = None
    st.session_state.last_detected = []
    st.session_state.last_latency = None

st.divider()

if scan_button:
    if source == "MacBook Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if not esp32_url:
            st.error("Please enter your ESP32 stream URL in the sidebar")
            st.stop()
        cap = cv2.VideoCapture(esp32_url)

    if not cap.isOpened():
        st.error("Could not open camera.")
        st.stop()

    # Set camera properties for better exposure
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # auto exposure

    # Warm up camera with 30 frames so exposure adjusts properly
    with st.spinner("Camera warming up..."):
        for _ in range(30):
            cap.read()
            time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Could not read frame")
        st.stop()

    # Check if frame is too dark and brighten it
    if frame.mean() < 30:
        frame = cv2.convertScaleAbs(frame, alpha=3.0, beta=50)
    elif frame.mean() < 60:
        frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=30)

    frame_height, frame_width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame = frame_rgb

    start_time = time.time()
    results = model(frame, verbose=False, conf=conf_threshold)
    names = model.names
    detected = []

    for r in results:
        for i, cls in enumerate(r.boxes.cls):
            label = names[int(cls)]
            box = r.boxes.xyxy[i].tolist()
            position = get_position(box, frame_width)
            entry = f"{label} — {position}"
            if entry not in detected:
                detected.append(entry)

    end_time = time.time()
    latency = round(end_time - start_time, 3)

    st.session_state.last_detected = detected
    st.session_state.last_latency = latency

    if detected:
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.history.insert(0, {
            "Time": timestamp,
            "Objects Detected": ", ".join(detected),
            "Latency": f"{latency}s"
        })
        st.session_state.history = st.session_state.history[:10]

        if speak_results:
            subprocess.run(["say", ", ".join(detected)])
    else:
        if speak_results:
            subprocess.run(["say", "Nothing detected"])

# always show last frame and results
if st.session_state.last_frame is not None:
    st.image(
        st.session_state.last_frame,
        caption="Last Captured Frame",
        width=640
    )

if st.session_state.last_detected:
    st.success("Detected: " + ", ".join(st.session_state.last_detected))
    st.caption(f"Inference latency: {st.session_state.last_latency}s")
elif st.session_state.last_latency is not None:
    st.warning("Nothing detected — try lowering confidence or improve lighting")

st.divider()
st.subheader("Detection History")
if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.caption("No scans yet — press Scan Now to start")