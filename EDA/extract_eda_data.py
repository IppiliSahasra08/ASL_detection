import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
DATASET_DIR = "Dataset_backup" 
OUTPUT_STATS_CSV = "video_stats.csv"
OUTPUT_COORDS_CSV = "wrist_coords.csv"
MODEL_PATH = "hand_landmarker.task" 

# Initialize Modern MediaPipe Hands (Tasks API)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

video_stats = []
wrist_coords = []

# Get list of classes (folders)
classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

for class_name in tqdm(classes, desc="Processing Classes"):
    class_dir = os.path.join(DATASET_DIR, class_name)
    videos = [v for v in os.listdir(class_dir) if v.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_name in videos:
        video_path = os.path.join(class_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        
        # 1. Basic Metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        brightness_list = []
        prev_x, prev_y = None, None
        total_movement = 0.0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 2. Environmental Quality (Brightness)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_list.append(np.mean(gray))
            
            # 3. Spatial & Kinematic (New MediaPipe API)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # The new API requires a specific mp.Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection_result = detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                # Grab the wrist landmark (landmark 0) of the first detected hand
                wrist = detection_result.hand_landmarks[0][0]
                h, w, _ = frame.shape
                curr_x, curr_y = int(wrist.x * w), int(wrist.y * h)
                
                # Save coordinates for the spatial heatmap
                wrist_coords.append({"class": class_name, "x": curr_x, "y": curr_y})
                
                # Calculate movement distance from previous frame
                if prev_x is not None and prev_y is not None:
                    distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    total_movement += distance
                    
                prev_x, prev_y = curr_x, curr_y
                
        cap.release()
        
        avg_brightness = np.mean(brightness_list) if brightness_list else 0
        
        video_stats.append({
            "class": class_name,
            "video_name": video_name,
            "frame_count": frame_count,
            "duration_sec": duration,
            "avg_brightness": avg_brightness,
            "total_movement_pixels": total_movement
        })

# Save to CSVs
pd.DataFrame(video_stats).to_csv(OUTPUT_STATS_CSV, index=False)
pd.DataFrame(wrist_coords).to_csv(OUTPUT_COORDS_CSV, index=False)

print(f"\nExtraction complete! Saved {OUTPUT_STATS_CSV} and {OUTPUT_COORDS_CSV}.")