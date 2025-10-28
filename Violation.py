import os
import cv2
from ultralytics import YOLO
import math 
import sys 
import tkinter as tk # For file dialog
from tkinter import filedialog # For file dialog

# --- Configuration ---
model_path = 'best.pt' # Path to your trained model file (ensure it's accessible)
confidence_threshold = 0.4 
correct_names = [
  'Number_plate', 'mobile_usage', 'pillion_rider_not_wearing_helmet', 
  'rider_and_pillion_not_wearing_helmet', 'rider_not_wearing_helmet', 
  'triple_riding', 'vehicle_with_offence' 
]
# --- ---

# Function to draw labels manually 
def draw_labels(frame, results):
    annotated_frame = frame.copy()
    boxes_data = results[0].boxes 

    if boxes_data is None:
        return annotated_frame

    boxes = boxes_data.xyxy.cpu().numpy().astype(int)
    confs = boxes_data.conf.cpu().numpy()              
    clss = boxes_data.cls.cpu().numpy().astype(int)    
    ids = boxes_data.id.cpu().numpy().astype(int) if boxes_data.id is not None else None

    for i in range(len(boxes)):
        box = boxes[i]
        conf = confs[i]
        cls_id = clss[i]
        track_id = ids[i] if ids is not None else None 

        x1, y1, x2, y2 = box
        
        try:
            class_name = correct_names[cls_id]
        except IndexError:
            class_name = f"UNK_{cls_id}"

        if track_id is not None:
            label = f"Id:{track_id} {class_name} {conf:.2f}" 
        else:
            label = f"{class_name} {conf:.2f}" 

        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        label_y = y1 - 2
        bg_y1 = y1 - h_text - 3
        if bg_y1 < 0:
           bg_y1 = y1 + 1 
           label_y = y1 + h_text + 1

        cv2.rectangle(annotated_frame, (x1, bg_y1), (x1 + w_text, label_y + 1), (0, 255, 0), -1) 
        cv2.putText(annotated_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA) 
            
    return annotated_frame

# --- Main Logic ---
cap = None 
try:
    # 1. Load Model
    model = YOLO(model_path)
    print("✅ Model loaded successfully.")

    # 2. Open File Dialog to Get Input Path
    root = tk.Tk() # Create a temporary hidden window
    root.withdraw() # Hide the main window
    input_path = filedialog.askopenfilename(
        title="Select Image or Video File",
        filetypes=[("Media Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.wmv"), 
                   ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                   ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                   ("All Files", "*.*")]
    )
    root.destroy() # Close the hidden window

    # 3. Check if a file was selected and exists
    if not input_path:
        print("No file selected. Exiting.")
        sys.exit() # Exit if user cancelled
        
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    print(f"Selected file: {input_path}")

    # 4. Determine if Image or Video
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    vid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    file_name, file_ext = os.path.splitext(input_path)
    file_ext = file_ext.lower()

    if file_ext in img_extensions:
        # --- Process Image ---
        print(f"Processing image...")
        
        frame = cv2.imread(input_path)
        if frame is None:
             raise Exception(f"Could not read image file: {input_path}")
        
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        
        annotated_frame_boxes = results[0].plot(labels=False, line_width=1)
        final_frame = draw_labels(annotated_frame_boxes, results)

        cv2.imshow('Image Detection Result (Press any key to close)', final_frame)
        print("Press any key in the image window to close.")
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

        print("\nDetected Violations:")
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            for cls_id, conf in zip(clss, confs):
                 try:
                     print(f"- {correct_names[cls_id]} (Confidence: {conf:.2f})")
                 except IndexError:
                     print(f"- UNKNOWN_CLASS_{cls_id} (Confidence: {conf:.2f})")
        else:
            print("No violations detected.")


    elif file_ext in vid_extensions:
        # --- Process Video ---
        print(f"Processing video...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")
        
        window_name = 'Video Violation Detection (Press Q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 

        print("Displaying video... Press 'q' in the window to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False)

            annotated_frame_boxes = results[0].plot(labels=False, line_width=1)
            final_frame = draw_labels(annotated_frame_boxes, results)

            cv2.imshow(window_name, final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                print("Quitting video processing...")
                break
        
        cap.release()
        cv2.destroyAllWindows()

    else:
        print(f"❌ Error: Unsupported file type '{file_ext}'. Please provide an image or video file.")

except FileNotFoundError as fnf_error:
     print(f"\n❌ File Error: {fnf_error}")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")

finally:
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("\nScript finished.")