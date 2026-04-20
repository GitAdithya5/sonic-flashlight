import streamlit as st
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="Sonic Flashlight",
    page_icon="🔦",
    layout="centered"
)

st.title("Sonic Flashlight")
st.caption("AI-powered assistive navigation device for the visually impaired")
st.divider()

st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.15,
    step=0.05
)
st.sidebar.divider()
st.sidebar.caption("Sonic Flashlight v1.0")
st.sidebar.caption("YOLO26n Nano Model")

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
    model = YOLO("yolo26n.pt")
    return model

with st.spinner("Loading YOLO26n model..."):
    model = load_model()

st.success("Model loaded — ready to scan")

def get_position(box, img_width):
    x1, y1, x2, y2 = box
    centre_x = (x1 + x2) / 2
    ratio = centre_x / img_width
    if ratio < 0.33:
        return "on your left"
    elif ratio > 0.66:
        return "on your right"
    else:
        return "in front of you"

st.subheader("Upload an image to scan")
st.caption("Take a photo on your phone and upload it here")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

col1, col2 = st.columns(2)
with col1:
    scan_button = st.button(
        "Scan Image",
        use_container_width=True,
        type="primary"
    )
with col2:
    clear_button = st.button(
        "Clear History",
        use_container_width=True
    )

if clear_button:
    st.session_state.history = []
    st.session_state.last_frame = None
    st.session_state.last_detected = []
    st.session_state.last_latency = None

st.divider()

if scan_button:
    if uploaded_file is None:
        st.warning("Please upload an image first")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_width = image.width

        st.session_state.last_frame = image

        start_time = time.time()
        results = model(img_array, verbose=False, conf=conf_threshold)
        names = model.names
        detected = []

        for r in results:
            for i, cls in enumerate(r.boxes.cls):
                label = names[int(cls)]
                box = r.boxes.xyxy[i].tolist()
                position = get_position(box, img_width)
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

if st.session_state.last_frame is not None:
    st.image(
        st.session_state.last_frame,
        caption="Scanned Image",
        width=640
    )

if st.session_state.last_detected:
    st.success("Detected: " + ", ".join(st.session_state.last_detected))
    st.caption(f"Inference latency: {st.session_state.last_latency}s")
elif st.session_state.last_latency is not None:
    st.warning("Nothing detected — try lowering confidence threshold")

st.divider()
st.subheader("Detection History")
if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.caption("No scans yet — upload an image and press Scan Image")