import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import collections

# Load model and encoder
print("Loading model...")
model = tf.keras.models.load_model('signspeak_model.h5')

with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

X_mean = np.load('X_mean.npy')
X_std = np.load('X_std.npy')

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Smoothing — use last 10 predictions
prediction_buffer = collections.deque(maxlen=10)

# Webcam
cap = cv2.VideoCapture(0)
print("✅ Real-time detection started. Press Q to quit.")

current_word = ""
confidence_score = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # Extract landmarks
        landmarks = result.multi_hand_landmarks[0]
        row = []
        for lm in landmarks.landmark:
            row += [lm.x, lm.y, lm.z]

        # Normalize
        row = np.array(row)
        row = (row - X_mean) / X_std
        row = row.reshape(1, -1)

        # Predict
        prediction = model.predict(row, verbose=0)
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        predicted_word = encoder.inverse_transform([predicted_class])[0]

        # Add to buffer
        prediction_buffer.append(predicted_word)

        # Only show if confidence > 70%
        if confidence > 0.70:
            # Most common prediction in buffer
            most_common = collections.Counter(
                prediction_buffer).most_common(1)[0][0]
            current_word = most_common
            confidence_score = confidence

    # Display
    h, w = frame.shape[:2]

    # Background box for text
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    if current_word:
        cv2.putText(frame, f"Sign: {current_word.upper()}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence_score*100:.1f}%",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Show your hand...",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.putText(frame, "Press Q to quit",
                (w - 180, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    cv2.imshow("SignSpeak - Real Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Session ended")