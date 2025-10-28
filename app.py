import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import math

# --- Configuration ---
MODEL_PATH = 'best.pt'  # Path to your model file relative to the script
CONFIDENCE_THRESHOLD = 0.4
CORRECT_NAMES = [
  'Number_plate', 'mobile_usage', 'pillion_rider_not_wearing_helmet', 
  'rider_and_pillion_not_wearing_helmet', 'rider_not_wearing_helmet', 
  'triple_riding', 'vehicle_with_offence' 
]
# --- ---

# --- Model Loading (Cached) ---
@st.cache_resource # Caches the model loading
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None

model = load_yolo_model(MODEL_PATH)

# --- Manual Label Drawing Function ---
def draw_labels(image_np, results):
    annotated_image = image_np.copy()
    boxes_data = results[0].boxes

    if boxes_data is None:
        return annotated_image

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
        if bg_y1 < 0:
            bg_y1 = y1 + 1
            label_y = y1 + h_text + 1

        cv2.rectangle(annotated_image, (x1, bg_y1), (x1 + w_text, label_y + 1), bg_color, -1)
        cv2.putText(annotated_image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        # Draw the bounding box itself
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), bg_color, thickness) # Use bg_color for box too

    return annotated_image

# --- Streamlit Interface ---
st.title("🚦 ML Powered Traffic Violation Detection")
st.write("Upload an image to detect traffic violations.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert('RGB') # Convert to RGB
    img_np = np.array(image) # Convert PIL Image to NumPy array (OpenCV format BGR)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV functions

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Processing...")

    try:
        # Run YOLO inference
        results = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False) # Use PIL image directly

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Draw labels manually on the BGR image
            annotated_image_bgr = draw_labels(img_bgr, results)
            # Convert BGR back to RGB for display in Streamlit
            annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

            st.image(annotated_image_rgb, caption='Processed Image with Detections.', use_column_width=True)

            # Print detected violations below the image
            st.write("Detected Violations:")
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            detected_any = False
            for cls_id, conf in zip(clss, confs):
                 try:
                     st.write(f"- {CORRECT_NAMES[cls_id]} (Confidence: {conf:.2f})")
                     detected_any = True
                 except IndexError:
                     st.write(f"- UNKNOWN_CLASS_{cls_id} (Confidence: {conf:.2f})")
                     detected_any = True
            if not detected_any:
                 st.write("No violations detected above the confidence threshold.")

        else:
            st.write("No violations detected.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        print(f"An error occurred during processing: {e}")

elif uploaded_file is None:
    st.write("Please upload an image file.")
else:
     st.error("Model could not be loaded. Cannot process the image.")

st.write("---")
st.write("Model: YOLOv8n Custom Trained")