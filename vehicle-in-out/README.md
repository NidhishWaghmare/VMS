Vehicle In/Out Detection
   This project uses YOLOv8 to detect vehicles (cars, buses, trucks) in RTSP video streams and classify their movement as "Entering" or "Exiting" across a user-defined gate line. It processes multiple camera streams, saves screenshots, and logs events in JSON files for real-time frontend updates.
Prerequisites

Python 3.8 or higher
Git
RTSP camera streams (configured in input.json)

Setup Instructions

Clone the Repository
git clone https://github.com/your-username/VMS.git
cd VMS/vehicle-in-out


Create a Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Run the Setup Script

Option 1: Using Python (recommended for all platforms)
python setup.py

This installs dependencies and downloads the YOLOv8 model (yolov8s.pt).

Option 2: Using Shell Script

On Windows:setup.bat


On Linux/macOS:chmod +x setup.sh
./setup.sh






Configure Cameras

Edit input.json to include your RTSP camera details. Example:[
    {
        "cameraId": "40731",
        "ip": "192.168.3.1",
        "rtspUrl": "rtsp://localhost:8554/mystream",
        "name": "",
        "location": null,
        "aiModels": [
            {
                "modelId": "1",
                "modelName": "vehicle in/out"
            }
        ]
    }
]




Run the Script
python vehicle_in_out.py


For each camera, a window will open to select a gate line by clicking two points.
Press q to skip a camera.
Screenshots are saved in data/screenshots/.
JSON logs are saved in data/logs/ (e.g., vehicle_log_40731.json).



Output

Screenshots: Saved in data/screenshots/ with filenames like <camera_id>_<event_type>_<timestamp>.jpg.
JSON Logs: Real-time logs in data/logs/ with entries like:{
    "timestamp": "2025-05-08 12:30:45",
    "event_type": "Entering",
    "entry_count": 1,
    "exit_count": 0,
    "screenshot_path": "data/screenshots/40731_Entering_2025-05-08_12-30-45.jpg",
    "camera_id": "40731",
    "location_id": "UNKNOWN",
    "model_id": "1",
    "model_name": "vehicle in/out"
}


Console: Prints final entry/exit counts for each camera.

Notes

Ensure RTSP URLs in input.json are accessible.
The script processes one vehicle per frame per camera. For multi-vehicle tracking, additional logic is needed.
Monitor JSON files in data/logs/ for real-time updates in a frontend application.

Troubleshooting

Model file missing: The setup script downloads yolov8s.pt automatically.
RTSP stream fails: Verify RTSP URLs and network connectivity.
Dependencies fail: Ensure Python version compatibility and try updating pip:pip install --upgrade pip



License
   MIT License
