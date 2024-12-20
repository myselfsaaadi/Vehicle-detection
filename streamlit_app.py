import streamlit as st
import cv2
import torch
import numpy as np
from comet_ml import Experiment
from ultralytics import YOLO  # Use ultralytics library for YOLOv5
import tempfile
from PIL import Image

# Initialize Comet Experiment
experiment = Experiment(
    api_key="9ehKWNPUhEaDwcQzNVh8zk1me",  # Replace with your Comet API key
    project_name="vehicle-detection-app",
    workspace="hihihoho"
)

# Load YOLOv5 model
@st.cache_resource
def load_model(model_path="yolov5x.pt"):
    return YOLO(model_path)  # Directly use the YOLO model from ultralytics

# Function to process video
def process_video(video_path, model, confidence_threshold, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Unable to open video file.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model.predict(frame, conf=confidence_threshold)
        detections = results[0].boxes.data.cpu().numpy()

        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            label = f"{model.names[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Log frame to Comet
        experiment.log_image(frame, name="Processed Frame")

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    return output_video_path

# Streamlit UI
st.title("YOLOv5x Vehicle Detection App")
st.sidebar.title("Settings")

# Confidence slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Video upload
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save the uploaded video to a temporary file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.sidebar.success("Video uploaded successfully!")

    # Load the YOLO model
    model = load_model("yolov5x.pt")

    # Process the video
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with st.spinner("Processing video..."):
        processed_video_path = process_video(temp_video_path, model, confidence_threshold, output_video_path)

    if processed_video_path:
        st.success("Video processing complete!")
        st.video(processed_video_path)  # Display the processed video
        experiment.log_asset(processed_video_path, asset_type="video", name="Processed Video")
