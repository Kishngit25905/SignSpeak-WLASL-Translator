import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pickle

print("Loading data...")
df = pd.read_csv('landmarks.csv')

# Features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Classes: {num_classes}")
print(f"Total samples: {len(X)}")

# Save encoder for later use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Normalize
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Save normalization values
np.save('X_mean.npy', X_mean)
np.save('X_std.npy', X_std)

# Build model
print("\nBuilding model...")
model = Sequential([
    Dense(512, activation='relu', input_shape=(63,)),
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
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'signspeak_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train
print("\nTraining started...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate
print("\nEvaluating...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}")

# Plot accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()
print("✅ Training graph saved as training_results.png")