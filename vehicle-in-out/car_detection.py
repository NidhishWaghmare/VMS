import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path
from datetime import datetime

# Configuration
SWAP_DIRECTIONS = False  # Set to True to swap "Entering" and "Exiting" labels
CROSSING_TOLERANCE = 10  # Pixels, tolerance for line crossing detection
MODEL_ID = "1"  # Model ID for vehicle in/out
MODEL_NAME = "vehicle in/out"

# Directories
BASE_DIR = Path(_file_).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "screenshots"
JSON_DIR = DATA_DIR / "logs"
MODEL_PATH = BASE_DIR / "yolov8s.pt"
INPUT_JSON = BASE_DIR / "input.json"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)

# Load YOLOv8 model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model file not found at {MODEL_PATH}. Please download yolov8s.pt and place it in {MODEL_PATH.parent}")
model = YOLO(str(MODEL_PATH))  # Using yolov8s.pt for vehicle detection

# Load camera configurations from input.json
if not INPUT_JSON.exists():
    raise FileNotFoundError(f"Input JSON file not found at {INPUT_JSON}")
with open(INPUT_JSON, 'r') as f:
    camera_configs = json.load(f)

# Filter cameras with modelId "1"
cameras = [cam for cam in camera_configs if any(model["modelId"] == MODEL_ID for model in cam["aiModels"])]
print(f"Found {len(cameras)} cameras with modelId '{MODEL_ID}': {cameras}")

# Global variables per camera
camera_states = {}

def initialize_camera_state():
    """Initialize state for a camera."""
    return {
        "roi_points": [],
        "roi_selected": False,
        "entry_count": 0,
        "exit_count": 0,
        "has_crossed": False,
        "vehicle_log": [],
        "prev_centroid": None,
        "cap": None,
        "width": None,
        "height": None,
        "first_frame": None,
        "line_params": None
    }

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select ROI line points."""
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
    return (a * x + b * y + c) / np.sqrt(a*2 + b*2)

def save_screenshot(frame, bbox, timestamp, event_type, camera_id):
    """Save a cropped screenshot of the vehicle and return its path."""
    x1, y1, x2, y2 = map(int, bbox)
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    vehicle_img = frame[y1:y2, x1:x2]
    
    timestamp_str = timestamp.replace(":", "-").replace(" ", "_")
    filename = f"{camera_id}{event_type}{timestamp_str}.jpg"
    filepath = OUTPUT_DIR / filename
    cv2.imwrite(str(filepath), vehicle_img)
    
    return str(filepath)

def update_json_log(camera_id, vehicle_log):
    """Update the JSON file with the current vehicle log for a camera."""
    json_output = JSON_DIR / f"vehicle_log_{camera_id}.json"
    with open(json_output, 'w') as f:
        json.dump(vehicle_log, f, indent=4)
    return str(json_output)

def determine_crossing(prev_centroid, curr_centroid, a, b, c, frame, bbox, camera_id, location, state):
    """Determine if the car crosses the ROI line and classify as Entering/Exiting."""
    if prev_centroid is None or curr_centroid is None:
        return None

    prev_x, prev_y = prev_centroid
    curr_x, curr_y = curr_centroid
    prev_dist = signed_distance(prev_x, prev_y, a, b, c)
    curr_dist = signed_distance(curr_x, curr_y, a, b, c)

    timestamp = datetime.now().strftime("%Y-%m-d %H:%M:%S")
    
    if not state["has_crossed"]:
        event_type = None
        if SWAP_DIRECTIONS:
            if prev_dist < -CROSSING_TOLERANCE and curr_dist >= -CROSSING_TOLERANCE:
                state["entry_count"] += 1
                state["has_crossed"] = True
                event_type = "Entering"
            elif prev_dist > CROSSING_TOLERANCE and curr_dist <= CROSSING_TOLERANCE:
                state["exit_count"] += 1
                state["has_crossed"] = True
                event_type = "Exiting"
        else:
            if prev_dist > CROSSING_TOLERANCE and curr_dist <= CROSSING_TOLERANCE:
                state["entry_count"] += 1
                state["has_crossed"] = True
                event_type = "Entering"
            elif prev_dist < -CROSSING_TOLERANCE and curr_dist >= -CROSSING_TOLERANCE:
                state["exit_count"] += 1
                state["has_crossed"] = True
                event_type = "Exiting"
        
        if event_type:
            filepath = save_screenshot(frame, bbox, timestamp, event_type, camera_id)
            
            log_entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "entry_count": state["entry_count"],
                "exit_count": state["exit_count"],
                "screenshot_path": filepath,
                "camera_id": camera_id,
                "location_id": location or "UNKNOWN",
                "model_id": MODEL_ID,
                "model_name": MODEL_NAME
            }
            
            state["vehicle_log"].append(log_entry)
            update_json_log(camera_id, state["vehicle_log"])
            
            return log_entry
    else:
        if abs(curr_dist) > CROSSING_TOLERANCE * 2:
            state["has_crossed"] = False
    return None

def initialize_cameras():
    """Initialize video streams and states for all cameras."""
    for camera in cameras:
        camera_id = camera["cameraId"]
        rtsp_url = camera["rtspUrl"]
        location = camera["location"]
        
        print(f"Initializing camera {camera_id} with video URL: {rtsp_url}")
        
        # Initialize state
        camera_states[camera_id] = initialize_camera_state()
        state = camera_states[camera_id]
        
        # Load video stream
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Failed to open video for camera {camera_id}")
            continue
        
        state["cap"] = cap
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Camera {camera_id} - Width: {width}, Height: {height}, FPS: {fps}")
        
        # Read first frame
        ret, first_frame = cap.read()
        if not ret or first_frame is None or first_frame.size == 0:
            print(f"Failed to read first frame for camera {camera_id}")
            cap.release()
            continue
        
        print(f"Camera {camera_id} - First frame shape: {first_frame.shape}")
        
        # Resize frame if resolution is low
        if width < 640 or height < 480:
            scale_factor = 2
            width = int(width * scale_factor)
            height = int(height * scale_factor)
        first_frame = cv2.resize(first_frame, (width, height))
        print(f"Camera {camera_id} - Resized frame shape: {first_frame.shape}")
        
        state["first_frame"] = first_frame
        state["width"] = width
        state["height"] = height
        
        # Set up window for ROI selection
        window_name = f"Select ROI Line - Camera {camera_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback, camera_id)

def process_roi_selection():
    """Handle ROI selection for all cameras simultaneously."""
    all_selected = False
    while not all_selected:
        all_selected = True
        for camera in cameras:
            camera_id = camera["cameraId"]
            state = camera_states.get(camera_id)
            if state is None or state["cap"] is None:
                continue
                
            if not state["roi_selected"]:
                all_selected = False
                temp_frame = state["first_frame"].copy()
                for pt in state["roi_points"]:
                    cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
                if len(state["roi_points"]) == 2:
                    cv2.line(temp_frame, state["roi_points"][0], state["roi_points"][1], (0, 0, 255), 2)
                cv2.putText(temp_frame, "Click two points to define gate line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(f"Select ROI Line - Camera {camera_id}", temp_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User aborted ROI selection")
            for camera in cameras:
                camera_id = camera["cameraId"]
                state = camera_states.get(camera_id)
                if state and state["cap"]:
                    state["cap"].release()
            cv2.destroyAllWindows()
            return False
    
    # After ROI selection, calculate line parameters
    for camera in cameras:
        camera_id = camera["cameraId"]
        state = camera_states.get(camera_id)
        if state is None or state["cap"] is None:
            continue
            
        x1, y1 = state["roi_points"][0]
        x2, y2 = state["roi_points"][1]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        state["line_params"] = (a, b, c)
        
        cv2.destroyWindow(f"Select ROI Line - Camera {camera_id}")
    
    return True

def process_frames():
    """Process frames for all cameras simultaneously."""
    frame_counts = {camera["cameraId"]: 0 for camera in cameras}
    active_cameras = len(cameras)
    
    # Create processing windows for each camera
    for camera in cameras:
        camera_id = camera["cameraId"]
        cv2.namedWindow(f"Camera {camera_id}", cv2.WINDOW_NORMAL)
    
    while active_cameras > 0:
        active_cameras = 0
        for camera in cameras:
            camera_id = camera["cameraId"]
            state = camera_states.get(camera_id)
            if state is None or state["cap"] is None:
                continue
                
            cap = state["cap"]
            if not cap.isOpened():
                continue
                
            active_cameras += 1
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {camera_id} - End of video or failed to read frame")
                cap.release()
                state["cap"] = None
                cv2.destroyWindow(f"Camera {camera_id}")
                continue
            
            frame_counts[camera_id] += 1
            location = camera["location"]
            
            frame = cv2.resize(frame, (state["width"], state["height"]))
            
            # Draw ROI line
            x1, y1 = state["roi_points"][0]
            x2, y2 = state["roi_points"][1]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Perform YOLO detection
            results = model(frame, classes=[2, 5, 7])  # Classes: 2=car, 5=bus, 7=truck
            detections = results[0].boxes.data.cpu().numpy()
            print(f"Camera {camera_id} - Frame {frame_counts[camera_id]}: {len(detections)} detections")
            
            current_centroid = None
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf < 0.25:
                    continue
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                current_centroid = calculate_centroid(bbox)
                
                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Check for crossing
                a, b, c = state["line_params"]
                log_entry = determine_crossing(state["prev_centroid"], current_centroid, a, b, c, frame, bbox, camera_id, location, state)
                
                break  # Process only the first detected vehicle
            
            state["prev_centroid"] = current_centroid
            
            # Display frame
            cv2.imshow(f"Camera {camera_id}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    for camera in cameras:
        camera_id = camera["cameraId"]
        state = camera_states.get(camera_id)
        if state and state["cap"]:
            print(f"Camera {camera_id} - Total Entries: {state['entry_count']}, Total Exits: {state['exit_count']}")
            update_json_log(camera_id, state["vehicle_log"])
            state["cap"].release()
    cv2.destroyAllWindows()

def main():
    """Run processing for all cameras simultaneously."""
    if not cameras:
        print("No cameras found with modelId '1'")
        return
    
    # Initialize all cameras
    initialize_cameras()
    
    # Handle ROI selection for all cameras
    if not process_roi_selection():
        return
    
    # Process frames for all cameras
    process_frames()

if _name_ == "_main_":
    main()