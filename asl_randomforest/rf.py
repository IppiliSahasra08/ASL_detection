import cv2
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
with open('sign_language_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded! Press Q to quit")

# Setup MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2  # CHANGED to 2
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        points = []
        # CHANGED: loop through all hands
        for hand_lms in result.hand_landmarks:
            for lm in hand_lms:
                points.extend([lm.x, lm.y, lm.z])

        # CHANGED: pad with zeros if only 1 hand
        if len(result.hand_landmarks) == 1:
            points.extend([0.0] * 63)

        # Draw dots on hands
        h, w, _ = frame.shape
        for hand_lms in result.hand_landmarks:
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Predict
        points = np.array(points).reshape(1, -1)
        prediction = model.predict(points)[0]
        confidence = max(model.predict_proba(points)[0]) * 100

        cv2.putText(frame, f"{prediction} ({confidence:.1f}%)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Show your hand!",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

    cv2.imshow('Sign Language Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()