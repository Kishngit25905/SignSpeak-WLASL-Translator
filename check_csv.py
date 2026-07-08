import pandas as pd

df = pd.read_csv('landmarks.csv')
counts = df['label'].value_counts().sort_values()

print(f"Total rows: {len(df)}")
print(f"Total words: {df['label'].nunique()}")
print(f"\nLowest counts:")
print(counts.head(15))
print(f"\nHighest counts:")
print(counts.tail(15))
print(f"\nAverage per word: {counts.mean():.1f}")
print(f"Min: {counts.min()}, Max: {counts.max()}")