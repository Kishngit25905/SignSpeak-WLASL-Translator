import json
import os
import shutil

# Load JSON
with open('WLASL_v0.3.json', 'r') as f:
    data = json.load(f)

# Check existing dataset
dataset_path = 'dataset'
word_counts = {}
for word in os.listdir(dataset_path):
    word_path = os.path.join(dataset_path, word)
    if os.path.isdir(word_path):
        count = len(os.listdir(word_path))
        word_counts[word] = count

print("Current word counts:")
weak_words = []
for word, count in sorted(word_counts.items(), key=lambda x: x[1]):
    status = "❌ WEAK" if count < 8 else "✅"
    print(f"  {word}: {count} videos {status}")
    if count < 8:
        weak_words.append(word)

print(f"\nWeak words to replace: {weak_words}")

# Find replacement words from dataset
print("\nFinding replacements...")
current_words = list(word_counts.keys())

replacements = []
for entry in data:
    word = entry['gloss']
    if word in current_words:
        continue
    
    # Count available videos
    available = sum(
        1 for inst in entry['instances']
        if os.path.exists(os.path.join('videos', f"{inst['video_id']}.mp4"))
    )
    
    if available >= 8:
        replacements.append((word, available))

replacements.sort(key=lambda x: x[1], reverse=True)
print(f"Available replacements: {replacements[:10]}")

# Remove weak words and add replacements
for i, weak_word in enumerate(weak_words):
    if i < len(replacements):
        new_word, count = replacements[i]
        
        # Remove weak word folder
        shutil.rmtree(os.path.join(dataset_path, weak_word))
        print(f"\n❌ Removed: {weak_word}")
        
        # Add replacement
        word_folder = os.path.join(dataset_path, new_word)
        os.makedirs(word_folder, exist_ok=True)
        
        for entry in data:
            if entry['gloss'] != new_word:
                continue
            for instance in entry['instances']:
                video_id = instance['video_id']
                src = os.path.join('videos', f'{video_id}.mp4')
                dst = os.path.join(word_folder, f'{video_id}.mp4')
                if os.path.exists(src):
                    shutil.copy2(src, dst)
        
        actual_count = len(os.listdir(word_folder))
        print(f"✅ Added: {new_word} ({actual_count} videos)")

# Final count
print("\n✅ Final dataset:")
total = 0
for word in sorted(os.listdir(dataset_path)):
    word_path = os.path.join(dataset_path, word)
    if os.path.isdir(word_path):
        count = len(os.listdir(word_path))
        total += count
        print(f"  {word}: {count} videos")
print(f"\nTotal videos: {total}")
print(f"Total words: {len(os.listdir(dataset_path))}")