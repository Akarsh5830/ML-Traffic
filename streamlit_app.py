import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import time # Included in the script logic

# --- Configuration ---
MODEL_PATH = 'best.pt'
CONFIDENCE_THRESHOLD = 0.4
CORRECT_NAMES = [
  'Number_plate', 'mobile_usage', 'pillion_rider_not_wearing_helmet',
  'rider_and_pillion_not_wearing_helmet', 'rider_not_wearing_helmet',
  'triple_riding', 'vehicle_with_offence'
]
# --- ---

# --- Model Loading (Cached) ---
@st.cache_resource
def load_yolo_model(model_path):
    try:
        if not os.path.exists(model_path):
             st.error(f"Model file not found at {model_path}. Please ensure 'best.pt' is in the app directory.")
             return None
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_yolo_model(MODEL_PATH)

# --- Manual Label Drawing Function ---
def draw_labels(frame, results):
    annotated_frame = frame.copy()
    boxes_data = results[0].boxes

    if boxes_data is None: return annotated_frame

    boxes = boxes_data.xyxy.cpu().numpy().astype(int)
    confs = boxes_data.conf.cpu().numpy()
    clss = boxes_data.cls.cpu().numpy().astype(int)
    ids = boxes_data.id.cpu().numpy().astype(int) if boxes_data.id is not None else None

    for i in range(len(boxes)):
        box, conf, cls_id = boxes[i], confs[i], clss[i]
        track_id = ids[i] if ids is not None else None

        x1, y1, x2, y2 = box

        try:
            class_name = CORRECT_NAMES[cls_id]
        except IndexError:
            class_name = f"UNK_{cls_id}"

        label = f"Id:{track_id} {class_name} {conf:.2f}" if track_id is not None else f"{class_name} {conf:.2f}"

        font_scale, thickness = 0.4, 1
        text_color, bg_color = (0, 0, 0), (0, 255, 0)

        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y, bg_y1 = y1 - 2, y1 - h_text - 3
        if bg_y1 < 0: bg_y1, label_y = y1 + 1, y1 + h_text + 1

        cv2.rectangle(annotated_frame, (x1, bg_y1), (x1 + w_text, label_y + 1), bg_color, -1)
        cv2.putText(annotated_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bg_color, thickness)

    return annotated_frame

# --- Streamlit Interface ---
st.set_page_config(layout="wide", page_title="Traffic Violation Detection")
st.title("🚦 ML Powered Traffic Violation Detection (Live Display)")
st.write("Upload an image or video for real-time (slow) processing and visualization.")
st.write("---")

uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None and model is not None:
    file_bytes = uploaded_file.read()
    is_image = uploaded_file.type.startswith('image/')
    is_video = uploaded_file.type.startswith('video/')

    col1, col2 = st.columns(2)

    with col2: # Display results in the second column
        st.subheader("Processed Output")
        process_placeholder = st.empty()
        st_summary = st.empty()
        progress_bar = st.progress(0.0)

    if is_image:
        with col1:
            st.subheader("Uploaded Image")
            st.image(file_bytes, use_column_width=True)
        # --- Image Processing Logic ---
        try:
            image = Image.open(uploaded_file).convert('RGB')
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            results = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                annotated_image_bgr = draw_labels(img_bgr, results)
                annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
                process_placeholder.image(annotated_image_rgb, caption='Processed Image.', use_column_width=True)

                # Print summary
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                detected_names = set(CORRECT_NAMES[i] for i in clss)
                
                summary_text = [f"**{name.replace('_', ' ').title()}**" for name in detected_names if name not in ['Number_plate', 'vehicle_with_offence']]
                
                st_summary.markdown("#### Detected Violations:")
                if summary_text:
                    st_summary.markdown(f"**Specific Violations:** {', '.join(summary_text)}")
                else:
                    st_summary.success("No specific rules broken above threshold.")
            else:
                process_placeholder.write("No violations detected above the threshold.")
            progress_bar.empty() # Clear the progress bar
        except Exception as e:
            st.error(f"Error processing image: {e}")
            progress_bar.empty()

    elif is_video:
        with col1:
            st.subheader("Uploaded Video")
            st.video(file_bytes)
            st.caption("Processing will be shown on the right.")

        # --- Video Processing Logic (Frame-by-Frame Display) ---
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(file_bytes)
        temp_video_path = tfile.name

        cap = None
        try:
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened(): raise IOError("Cannot open uploaded video file.")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0
            
            print("Starting video processing (live display)...")
            st_summary.warning("Processing in progress... This will be slower than real-time.")
            
            while True:
                ret, frame = cap.read()
                if not ret: break

                # Run tracking
                results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
                
                # Draw boxes and labels manually
                final_frame = draw_labels(frame, results)
                
                # *** Display the processed frame in Streamlit ***
                final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                process_placeholder.image(final_frame_rgb, caption=f'Processed Frame {processed_frames+1}/{frame_count if frame_count>0 else "Unknown"}', use_column_width=True)
                
                processed_frames += 1
                progress = min(1.0, processed_frames / frame_count if frame_count > 0 else 0)
                progress_bar.progress(progress)

            progress_bar.progress(1.0)
            st_summary.success("Video processing complete!")

        except Exception as e:
            st.error(f"Error processing video: {e}")
            print(f"Error processing video: {e}")
            st_summary.error("Processing failed.")
        finally:
            if cap is not None and cap.isOpened(): cap.release()
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    else:
        st.error("Unsupported file type uploaded.")

elif uploaded_file is None and model is not None:
    st.write("Waiting for file upload...")
else:
    st.error("Model not loaded. Cannot proceed.")

st.write("---")
st.caption("Developed using YOLOv8 and Streamlit.")


