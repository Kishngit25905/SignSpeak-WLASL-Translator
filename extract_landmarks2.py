import cv2
import mediapipe as mp
import numpy as np
import csv
import os

dataset_path = 'frames'
output_csv = 'landmarks.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

words = sorted(os.listdir(dataset_path))

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    header = []
    # Hand 1 (63) + Hand 2 (63) = 126 features
    for h in ['h1', 'h2']:
        for i in range(21):
            header += [f'{h}_x{i}', f'{h}_y{i}', f'{h}_z{i}']
    header.append('label')
    writer.writerow(header)

    total_written = 0
    total_skipped = 0

    for word in words:
        word_folder = os.path.join(dataset_path, word)
        if not os.path.isdir(word_folder):
            continue

        frames = os.listdir(word_folder)
        word_written = 0

        for frame_file in frames:
            frame_path = os.path.join(word_folder, frame_file)
            image = cv2.imread(frame_path)
            if image is None:
                total_skipped += 1
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                all_landmarks = result.multi_hand_landmarks
                row = []

                # Hand 1
                for lm in all_landmarks[0].landmark:
                    row += [lm.x, lm.y, lm.z]

                # Hand 2 (zeros if not detected)
                if len(all_landmarks) > 1:
                    for lm in all_landmarks[1].landmark:
                        row += [lm.x, lm.y, lm.z]
                else:
                    row += [0.0] * 63

                row.append(word)
                writer.writerow(row)
                word_written += 1
                total_written += 1
            else:
                total_skipped += 1

        print(f"✅ {word}: {word_written}/{len(frames)}")

print(f"\nTotal written: {total_written}")
print(f"Total skipped: {total_skipped}")
hands.close()