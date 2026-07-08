import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("Loading data...")
df = pd.read_csv('landmarks.csv')

X = df.drop('label', axis=1).values
y = df['label'].values

with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

y_encoded = encoder.transform(y)

X_mean = np.load('X_mean.npy')
X_std  = np.load('X_std.npy')

X = (X - X_mean) / X_std

_, X_test, _, y_test = train_test_split(
    X, y_encoded, test_size=0.2,
    random_state=42, stratify=y_encoded
)

print("Loading model...")
model = tf.keras.models.load_model('signspeak_model.h5')

print("Evaluating...")
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

# ── Classification Report ─────────────────────────────
labels = encoder.classes_
report = classification_report(y_test, y_pred, target_names=labels)
print("\n📊 Classification Report:")
print(report)

with open('classification_report.txt', 'w') as f:
    f.write(report)
print("✅ Saved: classification_report.txt")

# ── Confusion Matrix ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues')
plt.title('SignSpeak — Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("✅ Saved: confusion_matrix.png")

# ── Summary Stats ─────────────────────────────────────
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")
print(f"✅ Total test samples: {len(y_test)}")
print(f"✅ Total classes: {len(labels)}")