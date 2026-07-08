import json
import os
import shutil

# Our selected 50 words
SELECTED_WORDS = [
    'book', 'drink', 'computer', 'before', 'chair',
    'go', 'clothes', 'who', 'candy', 'cousin',
    'deaf', 'fine', 'help', 'no', 'thin',
    'walk', 'year', 'yes', 'all', 'black',
    'cool', 'finish', 'hot', 'like', 'many',
    'mother', 'now', 'orange', 'table', 'what',
    'woman', 'bed', 'blue', 'can', 'dog',
    'family', 'fish', 'hat', 'hearing', 'kiss',
    'language', 'later', 'man', 'shirt', 'study',
    'tall', 'white', 'wrong', 'apple', 'bird'
]

# Load JSON
with open('WLASL_v0.3.json', 'r') as f:
    data = json.load(f)

# Create output folder
os.makedirs('dataset', exist_ok=True)

total_copied = 0
missing = 0

print("Organizing dataset...\n")

for entry in data:
    word = entry['gloss']
    if word not in SELECTED_WORDS:
        continue

    # Create folder for this word
    word_folder = os.path.join('dataset', word)
    os.makedirs(word_folder, exist_ok=True)

    for instance in entry['instances']:
        video_id = instance['video_id']
        src = os.path.join('videos', f'{video_id}.mp4')
        dst = os.path.join('dataset', word, f'{video_id}.mp4')

        if os.path.exists(src):
            shutil.copy2(src, dst)
            total_copied += 1
        else:
            missing += 1

print(f"✅ Videos copied: {total_copied}")
print(f"⚠️  Missing videos: {missing}")
print(f"\nDataset folder structure:")
for word in SELECTED_WORDS:
    word_path = os.path.join('dataset', word)
    if os.path.exists(word_path):
        count = len(os.listdir(word_path))
        print(f"  {word}: {count} videos")