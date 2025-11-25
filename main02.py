import cv2
import numpy as np
import pywhatkit
import time
import os
# import win32clipboard
# -----------------------------------------------------------------------------
# --- 1. CONSTANTS & CONFIGURATION ---
# -----------------------------------------------------------------------------

# --- Model & Detection Parameters ---
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.4  # How sure the model needs to be (40%)
NMS_THRESHOLD = 0.45  # For filtering overlapping boxes

# --- Model Path ---
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the model file name
MODEL_FILE = "litter.onnx"  # Make sure this file is in the same folder
model_path = os.path.join(BASE_DIR, MODEL_FILE)

# --- Class Definitions (for yolov8n.onnx) ---
# All 80 classes your model knows
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- DEFINE YOUR TARGETS HERE ---
# These are the class names we consider "litter"
TARGET_CLASS_NAMES = [
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'
]

# This automatically converts our names to the correct Class IDs (e.g., 'bottle' becomes 39)
TARGET_CLASS_IDS = {i for i, name in enumerate(COCO_CLASSES) if name in TARGET_CLASS_NAMES}

# --- Alert Configuration ---
WHATSAPP_NUMBER = "+917007611944"  # !!! REPLACE WITH YOUR NUMBER (or group ID)
COOLDOWN_SECONDS = 100  # 5 minutes (to prevent spamming)
IMAGE_FILE_NAME = "alert_snapshot.jpg"
ALERT_COMBINED_IMAGE = "alert_snapshot_combo.jpg"  # combined annotated + close-up
IMAGE_JPEG_QUALITY = 95  # new: JPEG save quality (0-100)
FRAME_SCALE = 2  # scale factor for "little bigger" frame (1.0 = original size)

# -----------------------------------------------------------------------------
# --- 2. DETECTION FUNCTION ---
# -----------------------------------------------------------------------------

def detect_litter(frame, net):
    """
    Detects target objects in a frame, draws boxes, and returns:
      (litter_found: bool, annotated_frame: np.ndarray, detections: list)
    """
    # 1. --- PRE-PROCESSING ---
    original_height, original_width = frame.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(frame,
                                 scalefactor=1/255.0,
                                 size=(INPUT_WIDTH, INPUT_HEIGHT),
                                 swapRB=True,  # OpenCV uses BGR, model expects RGB
                                 crop=False)
    
    # 2. --- INFERENCE ---
    net.setInput(blob)
    try:
        outputs = net.forward()
    except Exception as e:
        print(f"Error during model inference: {e}")
        return False, frame, []  # <<< consistent 3-value return on error

    # 3. --- POST-PROCESSING ---
    outputs = np.transpose(np.squeeze(outputs[0])) # Shape is now (8400, 84)

    boxes = []
    confidences = []
    class_ids = []
    detections = []  # list to return

    x_scale = original_width / INPUT_WIDTH
    y_scale = original_height / INPUT_HEIGHT

    # Loop through all 8400 detections
    for row in outputs:
        class_scores = row[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        # 4. --- FILTERING ---
        if confidence > CONF_THRESHOLD and class_id in TARGET_CLASS_IDS:
            x_center, y_center, w, h = row[0:4]
            
            x_center_scaled = x_center * x_scale
            y_center_scaled = y_center * y_scale
            w_scaled = w * x_scale
            h_scaled = h * y_scale
            
            left = int(x_center_scaled - (w_scaled / 2))
            top = int(y_center_scaled - (h_scaled / 2))
            
            boxes.append([left, top, int(w_scaled), int(h_scaled)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # 5. --- NON-MAX SUPPRESSION (NMS) ---
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    litter_found = False
    if len(indices) > 0:
        litter_found = True  # We found at least one valid detection
        
        for i in indices.flatten():
            box = boxes[i]
            left, top, w, h = box

            # store detection for external use (crop, etc.)
            detections.append((left, top, w, h, class_ids[i], confidences[i]))

            class_name = COCO_CLASSES[class_ids[i]]
            conf_str = f"{confidences[i]*100:.1f}%"
            
            # --- DRAW ON FRAME ---
            cv2.rectangle(frame, (left, top), (left + w, top + h), (0, 0, 255), 2)
            label = f"{class_name}: {conf_str}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # now return detections as well
    return litter_found, frame, detections

# ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    # --- Load the model ---
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model from {model_path}")
        print(f"Error: {e}")
        print("Please make sure 'yolov8n.onnx' is in the same folder as this script.")
        return

    # --- Initialize Video Capture (ESP32-CAM HTTP stream) ---
    # The script will probe several common endpoints and try both default and FFMPEG backends.
    CANDIDATE_URLS = [
        "http://10.219.122.79/",
        # "http://10.149.210.79:81/stream",
        # "http://10.149.210.79:81/mjpeg",
        # "http://10.149.210.79:81/capture",
        # "http://10.149.210.79/stream", 
        # "http://10.149.210.79/capture",
        "http://10.219.122.79:81/",
        
    ]

    def try_open(url, backend=None, init_wait=0.5):
        try:
            cap = cv2.VideoCapture(url) if backend is None else cv2.VideoCapture(url, backend)
            time.sleep(init_wait)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            pass
        return None

    def open_stream_auto(urls, try_ffmpeg=True):
        # Try default backend first
        for u in urls:
            cap = try_open(u, backend=None)
            if cap:
                print(f"Opened stream with default backend: {u}")
                return cap, u, None
        # Then try FFMPEG backend if available
        if try_ffmpeg:
            for u in urls:
                try:
                    cap = try_open(u, backend=cv2.CAP_FFMPEG)
                except Exception:
                    cap = None
                if cap:
                    print(f"Opened stream with FFMPEG backend: {u}")
                    return cap, u, cv2.CAP_FFMPEG
        return None, None, None

    def reconnect_stream(current_url, current_backend):
        # Try reopening the same URL/backend first, then fallback to auto probe
        cap = try_open(current_url, backend=current_backend)
        if cap:
            return cap, current_url, current_backend
        return open_stream_auto(CANDIDATE_URLS, try_ffmpeg=True)

    cap, STREAM_URL, SELECTED_BACKEND = open_stream_auto(CANDIDATE_URLS, try_ffmpeg=True)
    if not cap:
        print("Warning: Could not open any ESP32 stream URL; falling back to local webcam.")
        cap = cv2.VideoCapture(0)
        STREAM_URL = None
        SELECTED_BACKEND = None
        if not cap.isOpened():
            print("Error: Could not open webcam either. Exiting.")
            return

    print("Starting litter detection... Press 'q' to quit.")

    # --- Initialize Alert Timer ---
    last_alert_time = 0

    while True:
        # Read a frame from the stream
        ret, frame = cap.read()
        if not ret or frame is None:
            # Try quick reconnect (do not crash)
            print("Warning: failed to read frame — attempting to reconnect stream...")
            try:
                cap.release()
            except Exception:
                pass
            cap, STREAM_URL, SELECTED_BACKEND = reconnect_stream(STREAM_URL, SELECTED_BACKEND)
            if not cap:
                print("Error: Reconnect failed; exiting.")
                break
            # try reading again
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Could not read frame after reconnect; skipping frame.")
                continue

        # --- Run Detection ---
        # This now returns: bool, frame-with-boxes, detections-list
        litter_was_detected, frame_with_boxes, detections = detect_litter(frame, net)
         
         
        # --- Trigger Alert ---
        if litter_was_detected:
            current_time = time.time()
            
            # This COOLDOWN is VITAL to avoid sending 1000 messages
            if (current_time - last_alert_time) > COOLDOWN_SECONDS:
                print("Litter detected! Sending alert...")
                
                # Save the frame *with the boxes on it* (use absolute paths)
                annotated_path = os.path.join(BASE_DIR, IMAGE_FILE_NAME)
                cv2.imwrite(annotated_path, frame_with_boxes) 
                
                # Create a combined image (annotated frame + cropped close-up of first detection)
                try:
                    if len(detections) > 0:
                        # pick first detection for close-up
                        left, top, w, h, cls_id, conf = detections[0]
                        left = max(0, left)
                        top = max(0, top)
                        right = min(left + w, frame_with_boxes.shape[1])
                        bottom = min(top + h, frame_with_boxes.shape[0])
                        crop = frame_with_boxes[top:bottom, left:right]

                        # If cropping failed (empty), create a white placeholder
                        if crop.size == 0:
                            crop = 255 * np.ones((frame_with_boxes.shape[0]//4, frame_with_boxes.shape[1]//6, 3), dtype=np.uint8)

                        # Resize both parts to a reasonable common height
                        target_h = min(frame_with_boxes.shape[0], 480)
                        scale = target_h / frame_with_boxes.shape[0]
                        annot_w = max(1, int(frame_with_boxes.shape[1] * scale))
                        annot_resized = cv2.resize(frame_with_boxes, (annot_w, target_h))
                        crop_w = max(1, int(annot_w * 0.28))
                        crop_resized = cv2.resize(crop, (crop_w, target_h))

                        combined = np.hstack([annot_resized, crop_resized])
                    else:
                        # fallback: just resize annotated frame
                        target_h = min(frame_with_boxes.shape[0], 480)
                        scale = target_h / frame_with_boxes.shape[0]
                        annot_w = max(1, int(frame_with_boxes.shape[1] * scale))
                        combined = cv2.resize(frame_with_boxes, (annot_w, target_h))

                    combined_path = os.path.join(BASE_DIR, ALERT_COMBINED_IMAGE)
                    cv2.imwrite(combined_path, combined)
                except Exception as e:
                    print(f"Failed to create combined alert image: {e}")
                    combined_path = annotated_path  # fallback to annotated only

                # Send the combined alert image (use absolute path)
                try:
                    pywhatkit.sendwhats_image(
                        WHATSAPP_NUMBER,
                        combined_path,
                        "Litter Detected from CV Project!"
                    )
                    last_alert_time = current_time # Reset cooldown
                    print("Alert sent. Cooldown started.")
                except Exception as e:
                    print(f"Failed to send WhatsApp alert: {e}")
                    # Don't reset cooldown if it failed, so it tries again
            else:
                # This just prints to your terminal, it doesn't send a message
                print("Litter detected, but in cooldown period.")

        # --- Show the video feed (now with boxes!) ---
        cv2.imshow("Litter Detection (Swachh Bharat Mission)", frame_with_boxes)

        # --- Quit if 'q' is pressed ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # --- Clean up ---
    cap.release()
    cv2.destroyAllWindows()

# --- This makes the script runnable ---
if __name__ == "__main__":
    main()