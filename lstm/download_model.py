import urllib.request
import os
print('Downloading hand landmarker model...')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_PATH = 'hand_landmarker.task'
urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
print(f'✓ Model downloaded to: {MODEL_PATH}')
print(f'Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB')
