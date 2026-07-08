import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('landmarks.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0) + 1e-8
X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std

np.save('X_mean.npy', X_mean)
np.save('X_std.npy',  X_std)

# ── Augmentation ──────────────────────────────────────
def augment(X, y, factor=3):
    X_aug, y_aug = [X], [y]
    for _ in range(factor):
        noise = np.random.normal(0, 0.02, X.shape)
        scale = np.random.uniform(0.95, 1.05, X.shape)
        X_aug.append(X * scale + noise)
        y_aug.append(y)
    return np.vstack(X_aug), np.concatenate(y_aug)

print("Augmenting data...")
X_train, y_train = augment(X_train, y_train, factor=4)

# Shuffle
idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]

print(f"After augmentation — Train: {len(X_train)}, Test: {len(X_test)}")

model = Sequential([
    Dense(1024, activation='relu', input_shape=(126,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
    ModelCheckpoint('signspeak_model.h5', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Final Test Accuracy: {accuracy*100:.2f}%")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_results.png')
print("✅ Saved training_results.png")