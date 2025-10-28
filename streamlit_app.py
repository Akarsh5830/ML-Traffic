import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import math
import tempfile 
import time 

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
        print("Model loaded successfully")
        # st.success("Model loaded successfully!") # Less intrusive message
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None

model = load_yolo_model(MODEL_PATH)

# --- Manual Label Drawing Function ---
# (Exactly the same function as before)
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
        
        try: class_name = CORRECT_NAMES[cls_id]
        except IndexError: class_name = f"UNK_{cls_id}"

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

st.title("🚦 ML Powered Traffic Violation Detection (Live Display Attempt)")
st.write("Upload an image or video. Video processing will be shown frame-by-frame (may be slow).")

uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    if model is None:
        st.error("Model failed to load. Cannot process file.")
    else:
        file_bytes = uploaded_file.read()
        is_image = uploaded_file.type.startswith('image/')
        is_video = uploaded_file.type.startswith('video/')

        st.write("---") # Separator

        if is_image:
            # --- Process Image ---
            col1, col2 = st.columns(2)
            with col1:
                st.image(file_bytes, caption='Uploaded Image.', use_column_width=True)
            with col2:
                st.write("Processing Image...")
                process_placeholder = st.empty()
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
                        # (Keep the violation printing loop from the previous script)
                        clss = results[0].boxes.cls.cpu().numpy().astype(int)
                        confs = results[0].boxes.conf.cpu().numpy()
                        for cls_id, conf in zip(clss, confs):
                             try: st.write(f"- {CORRECT_NAMES[cls_id]} (Confidence: {conf:.2f})")
                             except IndexError: st.write(f"- UNKNOWN_CLASS_{cls_id} (Confidence: {conf:.2f})")
                    else:
                        process_placeholder.write("No violations detected.")
                except Exception as e:
                    st.error(f"Error processing image: {e}")

        elif is_video:
            # --- Process Video (Frame-by-Frame Display) ---
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(file_bytes)
            temp_video_path = tfile.name
            
            st.write(f"Processing Video: {uploaded_file.name}")
            st.warning("⚠️ Displaying processed frames live. This will be slower than real-time playback.")
            
            # Placeholder for the video frame display
            st_frame_display = st.empty() 
            progress_bar = st.progress(0.0)
            
            cap = None
            try:
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened(): raise IOError("Cannot open uploaded video file.")

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processed_frames = 0
                
                print("Starting video processing (live display)...")

                while True:
                    ret, frame = cap.read()
                    if not ret: break # End of video

                    # Run tracking
                    results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    
                    # Draw boxes and labels manually
                    final_frame = draw_labels(frame, results) 
                    
                    # *** Display the processed frame in Streamlit ***
                    # Convert BGR (OpenCV) to RGB (Streamlit)
                    final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    st_frame_display.image(final_frame_rgb, caption=f'Processed Frame {processed_frames+1}/{frame_count if frame_count>0 else "Unknown"}', use_column_width=True)
                    
                    processed_frames += 1
                    progress = min(1.0, processed_frames / frame_count if frame_count > 0 else 0)
                    progress_bar.progress(progress)
                    # Add a small delay to allow Streamlit to update (optional)
                    # time.sleep(0.01) 

                print("Video processing finished.")
                progress_bar.progress(1.0)
                st.success("Video processing complete!")

            except Exception as e:
                st.error(f"Error processing video: {e}")
                print(f"Error processing video: {e}")
            finally:
                # Clean up resources
                if cap is not None and cap.isOpened(): cap.release()
                if 'temp_video_path' in locals() and os.path.exists(temp_video_path): 
                    try: os.remove(temp_video_path)
                    except Exception as e_clean: print(f"Error cleaning temp file: {e_clean}")
                cv2.destroyAllWindows() # Just in case

        else:
            st.error("Unsupported file type uploaded.")

elif uploaded_file is None:
    st.write("Waiting for file upload...")
else:
    st.error("Model not loaded. Cannot process file.")

st.write("---")
st.caption("Developed using YOLOv8 and Streamlit.")
