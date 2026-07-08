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
    max_num_hands=1,
    min_detection_confidence=0.5
)

words = sorted(os.listdir(dataset_path))
print(f"Processing {len(words)} words...\n")

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)

    # Header row
    header = []
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']
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
                landmarks = result.multi_hand_landmarks[0]
                row = []
                for lm in landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                row.append(word)
                writer.writerow(row)
                word_written += 1
                total_written += 1
            else:
                total_skipped += 1

        print(f"✅ {word}: {word_written}/{len(frames)} frames detected")

print(f"\n✅ Done!")
print(f"Total rows written to CSV: {total_written}")
print(f"Total frames skipped (no hand detected): {total_skipped}")
print(f"CSV saved as: {output_csv}")

hands.close()