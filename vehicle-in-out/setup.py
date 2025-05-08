import os
import subprocess
import urllib.request
from pathlib import Path

def create_directories():
    """Create required directories if they don't exist."""
    base_dir = Path(__file__).parent
    directories = [
        base_dir / "data" / "screenshots",
        base_dir / "data" / "logs",
        base_dir 
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print("Directories verified/created.")

def install_dependencies():
    """Install dependencies from requirements.txt."""
    requirements_path = Path(__file__).parent / "requirements.txt"
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements.txt not found at {requirements_path}")
    
    print("Installing dependencies...")
    try:
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        raise

def download_yolo_model():
    """Download yolov8s.pt if it doesn't exist."""
    model_dir = Path(__file__).parent 
    model_path = model_dir / "yolov8s.pt"
    
    if model_path.exists():
        print(f"YOLO model already exists at {model_path}")
        return
    
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    print(f"Downloading yolov8s.pt from {model_url}...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"YOLO model downloaded to {model_path}")
    except Exception as e:
        print(f"Failed to download yolov8s.pt: {e}")
        raise

def main():
    """Run setup tasks."""
    print("Starting setup for vehicle-in-out project...")
    create_directories()
    install_dependencies()
    download_yolo_model()
    print("Setup completed successfully!")

if __name__ == "__main__":
    main()