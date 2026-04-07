import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import urllib.request
from collections import Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TASK_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
if not os.path.exists(MODEL_TASK_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_TASK_PATH
    )


CNN_MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_model_1dcnn.pth')
METADATA_PATH = os.path.join(SCRIPT_DIR, 'landmarks_metadata.json')


class HandDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        
        features = []
        raw_landmarks = None
        
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            raw_landmarks = result.hand_landmarks
            # Get landmarks for up to 2 hands
            for hand in raw_landmarks[:2]:
                for lm in hand:
                    features.extend([lm.x, lm.y, lm.z])
            
            # If only one hand is detected, pad the rest with zeros (to reach 126)
            if len(raw_landmarks) == 1:
                features.extend([0.0] * 63)
        else:
            features = [0.0] * 126
            
        return features, raw_landmarks
    
    def close(self):
        self.detector.close()

# 1D CNN architecture
class SignLanguageCNN1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1) 

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.global_pool(self.relu3(self.bn3(self.conv3(x))))
        return self.classifier(x)

# main loop
def main():
    # metadata and labels
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    word_labels = list(metadata['word_to_idx'].keys())
    
    # load model
    device = torch.device('cpu')
    model = SignLanguageCNN1D(in_channels=126, num_classes=len(word_labels)).to(device)
    
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # init variables
    hand_detector = HandDetector()
    frame_buffer = []          
    prediction_history = []    
    sequence_length = 30
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Translator Active. Classes: {word_labels}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        features, raw_landmarks = hand_detector.detect(frame)
        
        if any(f != 0 for f in features):
            feat_arr = np.array(features, dtype=np.float32)
            wrist = feat_arr[:3]
            for i in range(42):
                feat_arr[i*3:(i+1)*3] -= wrist
            
            non_zero = feat_arr != 0
            if non_zero.any():
                mean, std = feat_arr[non_zero].mean(), feat_arr[non_zero].std()
                feat_arr[non_zero] = (feat_arr[non_zero] - mean) / (std + 1e-8)
            
            frame_buffer.append(feat_arr)
            if len(frame_buffer) > 60: 
                frame_buffer.pop(0)

        current_word = None
        current_conf = 0.0

        if len(frame_buffer) >= sequence_length:
            seq = np.array(frame_buffer[-sequence_length:])
            tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0) 
            tensor = tensor.permute(0, 2, 1) 
            
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                current_word = word_labels[pred_idx.item()]
                current_conf = conf.item()

        display_prediction = None
        display_confidence = 0.0

        if current_word:
            prediction_history.append((current_word, current_conf))
            if len(prediction_history) > 15:
                prediction_history.pop(0)

            counts = Counter([w for w, c in prediction_history])
            most_common, count = counts.most_common(1)[0]
            
            if count >= 3:
                avg_conf = np.mean([c for w, c in prediction_history if w == most_common])
                display_prediction = most_common
                display_confidence = avg_conf

        if raw_landmarks:
            for hand in raw_landmarks:
                for landmark in hand:
                    h, w = frame.shape[:2]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        h, w = frame.shape[:2]
        if display_prediction and display_confidence > 0.45:
            cv2.rectangle(frame, (20, 20), (550, 130), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (550, 130), (0, 255, 0), 2)
            cv2.putText(frame, display_prediction.upper(), (40, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"CONFIDENCE: {display_confidence:.1%}", (40, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # progress bar
        progress = min(len(frame_buffer) / sequence_length, 1.0)
        cv2.rectangle(frame, (20, h - 50), (200, h - 20), (30, 30, 30), -1)
        bar_color = (0, 255, 0) if progress >= 1.0 else (100, 100, 100)
        cv2.rectangle(frame, (20, h - 50), (20 + int(180 * progress), h - 20), bar_color, -1)
        cv2.putText(frame, f"BUFFER: {len(frame_buffer)}/{sequence_length}",
                    (210, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Sign Language Translator', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            frame_buffer = []
            prediction_history = []
    
    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()

if __name__ == '__main__':
    main()