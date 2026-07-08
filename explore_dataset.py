import json

# Load the main dataset JSON
with open('WLASL_v0.3.json', 'r') as f:
    data = json.load(f)

print(f"Total words in dataset: {len(data)}")
print("\nTop 60 words with most videos:\n")

word_counts = []
for entry in data:
    word = entry['gloss']
    video_count = len(entry['instances'])
    word_counts.append((word, video_count))

# Sort by video count descending
word_counts.sort(key=lambda x: x[1], reverse=True)

for word, count in word_counts[:60]:
    print(f"{word}: {count} videos")