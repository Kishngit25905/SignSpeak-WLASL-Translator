import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import collections
import time
import threading
import queue
import subprocess

# ── Speech ────────────────────────────────────────────
speech_queue = queue.Queue()

def speech_worker():
    while True:
        word = speech_queue.get()
        try:
            subprocess.run(
                ['powershell', '-Command',
                 f'Add-Type -AssemblyName System.Speech; '
                 f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                 f'$s.Rate = 1; '
                 f'$s.Speak("{word}")'],
                capture_output=True
            )
        except Exception as e:
            print(f"Speech error: {e}")
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(word):
    speech_queue.put(word)

# ── Load Model ────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model('signspeak_model.h5')

with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

X_mean = np.load('X_mean.npy')
X_std  = np.load('X_std.npy')

# ── MediaPipe ─────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── State ─────────────────────────────────────────────
HOLD_DURATION = 2.0
MIN_GAP       = 1.0

prediction_buffer = collections.deque(maxlen=15)
sentence          = []

current_word      = ""
confidence_score  = 0.0
hold_start        = None
holding_word      = ""
last_spoken_time  = 0.0
hold_progress     = 0.0

# ── Webcam ────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("✅ SignSpeak started! (Both hands supported)")
print("Hold any sign for 2 seconds to speak it.")
print("Press S to clear | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    detected_word = None

    if result.multi_hand_landmarks:
        for hl in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        all_hands = result.multi_hand_landmarks

        # Hand 1
        row = []
        for p in all_hands[0].landmark:
            row += [p.x, p.y, p.z]

        # Hand 2 (zeros if not present)
        if len(all_hands) > 1:
            for p in all_hands[1].landmark:
                row += [p.x, p.y, p.z]
        else:
            row += [0.0] * 63

        row  = (np.array(row) - X_mean) / X_std
        pred = model.predict(row.reshape(1, -1), verbose=0)
        conf = float(np.max(pred))
        cls  = int(np.argmax(pred))
        word = encoder.inverse_transform([cls])[0]

        prediction_buffer.append(word)

        if conf > 0.70:
            stable           = collections.Counter(
                prediction_buffer).most_common(1)[0][0]
            detected_word    = stable
            current_word     = stable
            confidence_score = conf

    # ── Hold Logic ────────────────────────────────────
    now = time.time()

    if detected_word:
        if detected_word != holding_word:
            holding_word  = detected_word
            hold_start    = now
            hold_progress = 0.0

        elapsed       = now - hold_start
        hold_progress = min(elapsed / HOLD_DURATION, 1.0)

        if elapsed >= HOLD_DURATION:
            if now - last_spoken_time >= MIN_GAP:
                print(f"🔊 Speaking: {holding_word}")
                speak(holding_word)
                sentence.append(holding_word)
                last_spoken_time = now
            holding_word  = ""
            hold_start    = None
            hold_progress = 0.0
    else:
        holding_word  = ""
        hold_start    = None
        hold_progress = 0.0

    # ── UI ────────────────────────────────────────────
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

    if current_word:
        cv2.putText(frame, f"Sign: {current_word.upper()}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence_score*100:.1f}%",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Show your hand...",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)

    hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    cv2.putText(frame, f"Hands: {hand_count}",
                (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    bar_w = int(w * hold_progress)
    cv2.rectangle(frame, (0, 88), (w, 95), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, 88), (bar_w, 95), (0, 255, 255), -1)

    cv2.rectangle(frame, (0, h - 55), (w, h), (0, 0, 0), -1)
    sentence_text = ' '.join(sentence[-7:])
    cv2.putText(frame, f"{sentence_text}",
                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 0), 2)
    cv2.putText(frame, "S=Clear  Q=Quit",
                (w - 170, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (180, 180, 180), 1)

    cv2.imshow("SignSpeak", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        sentence.clear()
        current_word = ""
        print("🗑️  Sentence cleared")

cap.release()
cv2.destroyAllWindows()
print("Final sentence:", ' '.join(sentence))
print("✅ Done")