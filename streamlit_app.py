import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import math
import tempfile # To handle temporary files
import time # To create unique filenames

# --- Configuration ---
MODEL_PATH = 'best.pt'  # Path to your model file relative to the script
CONFIDENCE_THRESHOLD = 0.4
CORRECT_NAMES = [
  'Number_plate', 'mobile_usage', 'pillion_rider_not_wearing_helmet',
  'rider_and_pillion_not_wearing_helmet', 'rider_not_wearing_helmet',
  'triple_riding', 'vehicle_with_offence'
]
TEMP_DIR = "temp_outputs" # Directory to store processed videos temporarily
os.makedirs(TEMP_DIR, exist_ok=True) # Create the temp dir if it doesn't exist
# --- ---

# --- Model Loading (Cached) ---
@st.cache_resource # Caches the model loading for efficiency
def load_yolo_model(model_path):
    try:
        if not os.path.exists(model_path):
             st.error(f"Model file not found at {model_path}. Please ensure 'best.pt' is in the app directory.")
             return None
        model = YOLO(model_path)
        print("Model loaded successfully")
        st.success("Model loaded successfully!") # Add success message
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None

model = load_yolo_model(MODEL_PATH)

# --- Manual Label Drawing Function ---
# (Exactly the same function as before - draws boxes and correct labels)
def draw_labels(frame, results):
    annotated_frame = frame.copy()
    boxes_data = results[0].boxes

    if boxes_data is None:
        return annotated_frame

    # Ensure tensors are on CPU and converted to numpy
    boxes = boxes_data.xyxy.cpu().numpy().astype(int)
    confs = boxes_data.conf.cpu().numpy()
    clss = boxes_data.cls.cpu().numpy().astype(int)
    # Tracking IDs might not be present in simple predict, handle None
    ids = boxes_data.id.cpu().numpy().astype(int) if boxes_data.id is not None else None

    for i in range(len(boxes)):
        box = boxes[i]
        conf = confs[i]
        cls_id = clss[i]
        track_id = ids[i] if ids is not None else None

        x1, y1, x2, y2 = box

        try:
            class_name = CORRECT_NAMES[cls_id]
        except IndexError:
            class_name = f"UNK_{cls_id}"

        if track_id is not None:
            label = f"Id:{track_id} {class_name} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"

        # Drawing settings
        font_scale = 0.4
        thickness = 1
        text_color = (0, 0, 0) # Black
        bg_color = (0, 255, 0) # Green

        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = y1 - 2
        bg_y1 = y1 - h_text - 3
        if bg_y1 < 0: # Adjust if label goes off screen top
            bg_y1 = y1 + 1
            label_y = y1 + h_text + 1

        cv2.rectangle(annotated_frame, (x1, bg_y1), (x1 + w_text, label_y + 1), bg_color, -1)
        cv2.putText(annotated_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        # Draw the bounding box itself
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bg_color, thickness) # Use bg_color for box too

    return annotated_frame

# --- Streamlit Interface ---
st.set_page_config(layout="wide", page_title="Traffic Violation Detection") # Use wide layout

st.title("🚦 ML Powered Traffic Violation Detection")
st.write("Upload an image or video to detect traffic violations using a custom YOLOv8 model.")

uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Check if model loaded successfully before proceeding
    if model is None:
        st.error("Model failed to load. Cannot process file.")
    else:
        file_bytes = uploaded_file.read()
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}

        # Determine file type
        is_image = uploaded_file.type.startswith('image/')
        is_video = uploaded_file.type.startswith('video/')

        col1, col2 = st.columns(2) # Create two columns for display

        with col1:
            if is_image:
                 st.image(file_bytes, caption='Uploaded Image.', use_column_width=True)
            elif is_video:
                 # Displaying uploaded video can be resource intensive, optional
                 st.video(file_bytes)
                 st.caption('Uploaded Video.')

        with col2:
            st.write("Processing...")
            process_placeholder = st.empty() # Placeholder for status/results

            if is_image:
                # --- Process Image ---
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    img_np = np.array(image)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

                    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                        annotated_image_bgr = draw_labels(img_bgr, results)
                        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
                        process_placeholder.image(annotated_image_rgb, caption='Processed Image.', use_column_width=True)

                        # Print detected violations
                        st.write("Detected Violations:")
                        clss = results[0].boxes.cls.cpu().numpy().astype(int)
                        confs = results[0].boxes.conf.cpu().numpy()
                        for cls_id, conf in zip(clss, confs):
                             try:
                                 st.write(f"- {CORRECT_NAMES[cls_id]} (Confidence: {conf:.2f})")
                             except IndexError:
                                 st.write(f"- UNKNOWN_CLASS_{cls_id} (Confidence: {conf:.2f})")

                    else:
                        process_placeholder.write("No violations detected above the confidence threshold.")

                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    print(f"Error processing image: {e}")

            elif is_video:
                # --- Process Video ---
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(file_bytes)
                temp_video_path = tfile.name
                
                # Generate a unique output filename using timestamp
                timestamp = int(time.time())
                safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in (' ', '.', '_')).rstrip()
                output_video_path = os.path.join(TEMP_DIR, f"output_{timestamp}_{safe_filename}.avi")

                cap = None
                video_writer = None
                try:
                    cap = cv2.VideoCapture(temp_video_path)
                    if not cap.isOpened():
                        raise IOError("Cannot open uploaded video file.")

                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                    if not video_writer.isOpened():
                        raise Exception("Error: Could not open video writer.")

                    process_placeholder.text("Processing video frame by frame...")
                    progress_bar = st.progress(0.0)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    processed_frames = 0
                    
                    print("Starting video processing...")

                    while True:
                        ret, frame = cap.read()
                        if not ret: break # End of video

                        results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
                        final_frame = draw_labels(frame, results) # Draw on original frame
                        video_writer.write(final_frame)

                        processed_frames += 1
                        progress = min(1.0, processed_frames / frame_count if frame_count > 0 else 0)
                        progress_bar.progress(progress)

                    print("Video processing finished.")
                    progress_bar.progress(1.0) # Ensure bar is full

                    cap.release()
                    video_writer.release()
                    process_placeholder.success("Video processing complete!")

                    # Provide download link
                    with open(output_video_path, "rb") as file_dl:
                        st.download_button(
                            label="Download Processed Video (.avi)",
                            data=file_dl,
                            file_name=os.path.basename(output_video_path), # Use just the filename
                            mime="video/avi"
                        )
                    # Clean up temporary input file
                    os.remove(temp_video_path)
                    # Note: Output file remains for download

                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    print(f"Error processing video: {e}")
                    # Clean up resources if error occurs
                    if cap is not None and cap.isOpened(): cap.release()
                    if video_writer is not None: video_writer.release() # Check if initialized
                    if 'temp_video_path' in locals() and os.path.exists(temp_video_path): os.remove(temp_video_path)

            else:
                st.error("Unsupported file type uploaded.")

elif uploaded_file is None:
    st.write("Waiting for file upload...")
else: # Handles the case where model loading failed earlier
    st.error("Model not loaded. Cannot process file.")

st.write("---")
st.caption("Developed using YOLOv8 and Streamlit.")
