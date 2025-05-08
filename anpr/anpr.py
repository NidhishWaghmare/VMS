import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import threading
from pathlib import Path
from datetime import datetime

# Configuration
CROSSING_TOLERANCE = 10  # Pixels, tolerance for line crossing detection
MODEL_ID = "2"  # Model ID for ANPR
MODEL_NAME = "ANPR"

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "screenshots"
JSON_DIR = DATA_DIR / "logs"
INPUT_JSON = BASE_DIR / "input.json"
VEHICLE_MODEL_PATH = BASE_DIR / "models" / "yolov8s.pt"
ANPR_MODEL_PATH = BASE_DIR / "models" / "ANPR.pt"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)

# Load YOLOv8 model for vehicle detection
if not VEHICLE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Vehicle model file not found at {VEHICLE_MODEL_PATH}")
vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))

# Load ANPR model for number plate detection
if not ANPR_MODEL_PATH.exists():
    raise FileNotFoundError(f"ANPR model file not found at {ANPR_MODEL_PATH}")
anpr_model = YOLO(str(ANPR_MODEL_PATH))

# Global variables for ROI line selection per camera
camera_states = {}

def initialize_camera_state():
    """Initialize state for a camera."""
    return {
        "roi_points": [],
        "roi_selected": False,
        "has_crossed": False,
        "prev_centroid": None
    }

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select ROI line points for a specific camera."""
    camera_id = param
    state = camera_states[camera_id]
    if event == cv2.EVENT_LBUTTONDOWN and len(state["roi_points"]) < 2:
        state["roi_points"].append((x, y))
        if len(state["roi_points"]) == 2:
            state["roi_selected"] = True

def calculate_centroid(bbox):
    """Calculate the centroid of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def signed_distance(x, y, a, b, c):
    """Calculate signed distance from point (x, y) to line ax + by + c = 0."""
    return (a * x + b * y + c) / np.sqrt(a**2 + b**2)

def save_screenshot(frame, bbox, timestamp, prefix, camera_id):
    """Save a cropped screenshot and return its path."""
    x1, y1, x2, y2 = map(int, bbox)
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    img = frame[y1:y2, x1:x2]
    
    timestamp_str = timestamp.replace(":", "-").replace(" ", "_")
    filename = f"{prefix}_{camera_id}_{timestamp_str}.jpg"
    filepath = OUTPUT_DIR / filename
    cv2.imwrite(str(filepath), img)
    
    return str(filepath)

def detect_number_plate(frame, bbox):
    """Detect number plate in the cropped vehicle region."""
    x1, y1, x2, y2 = map(int, bbox)
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    vehicle_img = frame[y1:y2, x1:x2]
    
    results = anpr_model(vehicle_img)
    detections = results[0].boxes.data.cpu().numpy()
    
    if len(detections) > 0:
        best_det = max(detections, key=lambda x: x[4])
        if best_det[4] > 0.6:
            plate_x1, plate_y1, plate_x2, plate_y2 = map(int, best_det[:4])
            plate_x1 += x1
            plate_y1 += y1
            plate_x2 += x1
            plate_y2 += y1
            return [plate_x1, plate_y1, plate_x2, plate_y2], vehicle_img[plate_y1-y1:plate_y2-y1, plate_x1-x1:plate_x2-x1]
    return None, None

def determine_crossing(prev_centroid, curr_centroid, a, b, c, frame, bbox, camera_id, camera_info, state):
    """Determine if the car crosses the ROI line and log to JSON."""
    if prev_centroid is None or curr_centroid is None:
        return None

    prev_x, prev_y = prev_centroid
    curr_x, curr_y = curr_centroid
    prev_dist = signed_distance(prev_x, prev_y, a, b, c)
    curr_dist = signed_distance(curr_x, curr_y, a, b, c)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = None
    
    if not state["has_crossed"]:
        if prev_dist > CROSSING_TOLERANCE and curr_dist <= CROSSING_TOLERANCE:
            state["has_crossed"] = True
            vehicle_filepath = save_screenshot(frame, bbox, timestamp, "vehicle_Entering", camera_id)
            plate_bbox, _ = detect_number_plate(frame, bbox)
            plate_filepath = None
            if plate_bbox:
                plate_filepath = save_screenshot(frame, plate_bbox, timestamp, "plate_Entering", camera_id)
            
            log_entry = {
                "timestamp": timestamp,
                "vehicle_screenshot": vehicle_filepath,
                "plate_screenshot": plate_filepath,
                "cameraId": camera_id,
                "location": camera_info.get("location", None),
                "modelId": MODEL_ID,
                "modelName": MODEL_NAME,
                "status": "Entering"
            }
        elif prev_dist < -CROSSING_TOLERANCE and curr_dist >= -CROSSING_TOLERANCE:
            state["has_crossed"] = True
            vehicle_filepath = save_screenshot(frame, bbox, timestamp, "vehicle_Exiting", camera_id)
            
            log_entry = {
                "timestamp": timestamp,
                "vehicle_screenshot": vehicle_filepath,
                "plate_screenshot": None,
                "cameraId": camera_id,
                "location": camera_info.get("location", None),
                "modelId": MODEL_ID,
                "modelName": MODEL_NAME,
                "status": "Exiting"
            }
    else:
        if abs(curr_dist) > CROSSING_TOLERANCE * 2:
            state["has_crossed"] = False
    
    return log_entry

def append_to_json_log(log_entry, camera_id):
    """Append log entry to the JSON file for the camera."""
    json_filename = f"anpr_logs_{camera_id}.json"
    json_filepath = JSON_DIR / json_filename
    
    if json_filepath.exists():
        with open(json_filepath, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(json_filepath, 'w') as f:
        json.dump(logs, f, indent=4)

def process_camera(camera):
    """Process video stream for a single camera."""
    camera_id = camera["cameraId"]
    video_path = camera["rtspUrl"]
    
    # Initialize state
    camera_states[camera_id] = initialize_camera_state()
    state = camera_states[camera_id]
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open stream for camera {camera_id}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize frame if resolution is low
    if width < 640 or height < 480:
        scale_factor = 2
        width = int(width * scale_factor)
        height = int(height * scale_factor)
    
    # Read first frame for ROI selection
    ret, first_frame = cap.read()
    if not ret:
        print(f"Failed to read first frame for camera {camera_id}")
        cap.release()
        return
    
    first_frame = cv2.resize(first_frame, (width, height))
    
    # Set up window for ROI selection
    window_name = f"Select ROI Line - {camera_id}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param=camera_id)
    
    # Display first frame and wait for ROI selection
    while not state["roi_selected"]:
        temp_frame = first_frame.copy()
        for pt in state["roi_points"]:
            cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
        if len(state["roi_points"]) == 2:
            cv2.line(temp_frame, state["roi_points"][0], state["roi_points"][1], (0, 0, 255), 2)
        cv2.putText(temp_frame, "Click two points to define gate line", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord('r'):
            state["roi_points"] = []
            state["roi_selected"] = False
    
    cv2.destroyWindow(window_name)
    
    # Define ROI line
    x1, y1 = state["roi_points"][0]
    x2, y2 = state["roi_points"][1]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (width, height))
        
        # Perform vehicle detection
        results = vehicle_model(frame, classes=[2, 5, 7])
        detections = results[0].boxes.data.cpu().numpy()
        
        current_centroid = None
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf < 0.25:
                continue
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            centroid = calculate_centroid(bbox)
            current_centroid = centroid
            
            # Determine if the car crosses the ROI line
            log_entry = determine_crossing(state["prev_centroid"], current_centroid, a, b, c, frame, bbox, camera_id, camera, state)
            if log_entry:
                append_to_json_log(log_entry, camera_id)
            
            break
        
        state["prev_centroid"] = current_centroid
    
    cap.release()

def main():
    """Run processing for all cameras in parallel."""
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Input JSON file not found at {INPUT_JSON}")
    
    # Load input JSON
    with open(INPUT_JSON, 'r') as f:
        cameras = json.load(f)
    
    # Filter cameras with modelId = "2" (ANPR)
    anpr_cameras = [
        cam for cam in cameras
        if any(model["modelId"] == MODEL_ID for model in cam["aiModels"])
    ]
    
    if not anpr_cameras:
        print("No cameras found with ANPR model (modelId = 2)")
        return
    
    threads = []
    for camera in anpr_cameras:
        thread = threading.Thread(target=process_camera, args=(camera,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()