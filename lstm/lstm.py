""""""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import urllib.request
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print('Downloading hand landmarker model...')
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', MODEL_PATH)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
class HandDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, min_hand_detection_confidence=0.7, running_mode=vision.RunningMode.IMAGE)
        self.detector = vision.HandLandmarker.create_from_options(options)
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        features = []
        hand_landmarks = None
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_landmarks = result.hand_landmarks
            for hand in hand_landmarks[:2]:
                for lm in hand:
                    features.extend([lm.x, lm.y, lm.z])
            if len(result.hand_landmarks) == 1:
                features.extend([0.0] * 63)
        else:
            features = [0.0] * 126
        return (features, hand_landmarks)
    def close(self):
        self.detector.close()
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Softmax(dim=1))
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)
        return (weighted_output, attention_weights)
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        self._init_weights()
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return (logits, attention_weights)
def main():
    model_path = 'validation_lstm/best_model_finetuned.pth'
    metadata_path = 'landmarks_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    word_labels = list(metadata['word_to_idx'].keys())
    num_classes = len(word_labels)
    device = torch.device('cpu')
    model = SignLanguageLSTM(input_size=126, hidden_size=256, num_layers=2, num_classes=num_classes, dropout=0.0).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    hand_detector = HandDetector()
    frame_buffer = []
    prediction_history = []
    sequence_length = 30
    print(f'\nClasses: {word_labels}')
    print("Press 'Q' to quit, 'C' to clear buffer\n")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks, hand_landmarks = hand_detector.detect(frame)
        if any((l != 0 for l in landmarks)):
            landmarks = np.array(landmarks, dtype=np.float32)
            wrist = landmarks[:3]
            for i in range(42):
                landmarks[i * 3:(i + 1) * 3] -= wrist
            non_zero = landmarks != 0
            if non_zero.any():
                mean = landmarks[non_zero].mean()
                std = landmarks[non_zero].std()
                if std > 0:
                    landmarks[non_zero] = (landmarks[non_zero] - mean) / (std + 1e-08)
            frame_buffer.append(landmarks.tolist())
            if len(frame_buffer) > 60:
                frame_buffer.pop(0)
        prediction = None
        confidence = 0.0
        if len(frame_buffer) >= sequence_length:
            sequence = np.array(frame_buffer[-sequence_length:])
            tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs, _ = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                prediction = word_labels[pred_idx.item()]
                confidence = conf.item()
        if prediction:
            prediction_history.append((prediction, confidence))
            if len(prediction_history) > 15:
                prediction_history.pop(0)
            word_counts = {}
            for w, c in prediction_history:
                word_counts[w] = word_counts.get(w, 0) + 1
            most_common = max(word_counts.items(), key=lambda x: x[1])
            if most_common[1] >= 3:
                avg_conf = np.mean([c for w, c in prediction_history if w == most_common[0]])
                prediction = most_common[0]
                confidence = avg_conf
        if hand_landmarks:
            for hand in hand_landmarks:
                for landmark in hand:
                    h, w = frame.shape[:2]
                    x, y = (int(landmark.x * w), int(landmark.y * h))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                for i in range(len(hand) - 1):
                    h, w = frame.shape[:2]
                    x1, y1 = (int(hand[i].x * w), int(hand[i].y * h))
                    x2, y2 = (int(hand[i + 1].x * w), int(hand[i + 1].y * h))
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        h, w = frame.shape[:2]
        if prediction and confidence > 0.5:
            cv2.rectangle(frame, (20, 20), (550, 130), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (550, 130), (0, 255, 0), 2)
            cv2.putText(frame, prediction.upper(), (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f'{confidence:.1%}', (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        progress = min(len(frame_buffer) / sequence_length, 1.0)
        cv2.rectangle(frame, (20, h - 50), (200, h - 20), (30, 30, 30), -1)
        color = (0, 255, 0) if progress >= 1.0 else (100, 100, 100)
        cv2.rectangle(frame, (20, h - 50), (20 + int(180 * progress), h - 20), color, -1)
        cv2.putText(frame, f'{len(frame_buffer)}/{sequence_length}', (210, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, 'Q=Quit C=Clear', (w - 180, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('Sign Language Translator', frame)
        key = cv2.waitKey(1) & 255
        if key == ord('q'):
            break
        elif key == ord('c'):
            frame_buffer = []
            prediction_history = []
    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()
    print('\nDone!')
if __name__ == '__main__':
    main()
