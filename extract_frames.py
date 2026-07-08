import cv2
import os

dataset_path = 'dataset'
frames_path = 'frames'
os.makedirs(frames_path, exist_ok=True)

FRAMES_PER_VIDEO = 30  # Extract 30 frames per video

words = sorted(os.listdir(dataset_path))
print(f"Processing {len(words)} words...\n")

total_videos = 0
total_frames = 0

for word in words:
    word_folder = os.path.join(dataset_path, word)
    if not os.path.isdir(word_folder):
        continue

    frames_word_folder = os.path.join(frames_path, word)
    os.makedirs(frames_word_folder, exist_ok=True)

    videos = [v for v in os.listdir(word_folder) if v.endswith('.mp4')]

    for video_file in videos:
        video_path = os.path.join(word_folder, video_file)
        video_id = video_file.replace('.mp4', '')

        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_video_frames == 0:
            cap.release()
            continue

        # Pick evenly spaced frame indices
        indices = [int(i * total_video_frames / FRAMES_PER_VIDEO)
                   for i in range(FRAMES_PER_VIDEO)]

        frame_count = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count in indices:
                frame_filename = os.path.join(
                    frames_word_folder,
                    f"{video_id}_frame{saved:03d}.jpg"
                )
                frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(frame_filename, frame)
                saved += 1
            frame_count += 1

        cap.release()
        total_frames += saved
        total_videos += 1

    print(f"✅ {word}: {len(videos)} videos → {len(os.listdir(frames_word_folder))} frames")

print(f"\n✅ Done!")
print(f"Total videos processed: {total_videos}")
print(f"Total frames extracted: {total_frames}")