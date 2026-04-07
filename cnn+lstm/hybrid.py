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

# config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_TASK_PATH = os.path.join(BASE_DIR, 'cnn', "hand_landmarker.task")

# Ensure the cnn directory exists if we need to download it there
os.makedirs(os.path.dirname(MODEL_TASK_PATH), exist_ok=True)

if not os.path.exists(MODEL_TASK_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_TASK_PATH
    )

HYBRID_MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_model_hybrid.pth')
METADATA_PATH = os.path.join(BASE_DIR, 'cnn', 'landmarks_metadata.json')


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
            for hand in raw_landmarks[:2]:
                for lm in hand:
                    features.extend([lm.x, lm.y, lm.z])
            
            if len(raw_landmarks) == 1:
                features.extend([0.0] * 63) 
        else:
            features = [0.0] * 126
            
        return features, raw_landmarks
    
    def close(self):
        self.detector.close()


class SignLanguageHybrid(nn.Module):
    def __init__(self, num_classes, input_dim=126, hidden_dim=256, num_lstm_layers=2):
        super(SignLanguageHybrid, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=num_lstm_layers, 
            batch_first=True
        )
        
    
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])

# main loop
def main():
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    word_labels = list(metadata['word_to_idx'].keys())

    device = torch.device('cpu')
    model = SignLanguageHybrid(num_classes=len(word_labels)).to(device)
    
    checkpoint = torch.load(HYBRID_MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    hand_detector = HandDetector()
    frame_buffer = []          
    prediction_history = []    
    sequence_length = 30
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Hybrid Translator Active. Classes: {word_labels}")

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
            if len(frame_buffer) > 45: 
                frame_buffer.pop(0)

        current_word = None
        current_conf = 0.0

        if len(frame_buffer) >= sequence_length:
            seq = np.array(frame_buffer[-sequence_length:])
            tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0) # (1, 30, 126)
            
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
            
            if count >= 5: # Require more agreement for LSTM stability
                avg_conf = np.mean([c for w, c in prediction_history if w == most_common])
                display_prediction = most_common
                display_confidence = avg_conf

        if raw_landmarks:
            for hand in raw_landmarks:
                for landmark in hand:
                    h, w = frame.shape[:2]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        h, w = frame.shape[:2]
        if display_prediction and display_confidence > 0.3:
            cv2.rectangle(frame, (20, 20), (550, 140), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (550, 140), (0, 255, 255), 2)
            cv2.putText(frame, display_prediction.upper(), (40, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
            cv2.putText(frame, f"HYBRID CONF: {display_confidence:.1%}", (40, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        progress = min(len(frame_buffer) / sequence_length, 1.0)
        cv2.rectangle(frame, (20, h-40), (220, h-20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h-40), (20 + int(200 * progress), h-20), (0, 255, 0), -1)
        cv2.putText(frame, "SEQUENCE READY" if progress >= 1.0 else "COLLECTING...", 
                    (230, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Hybrid Sign Language Translator', frame)
        
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