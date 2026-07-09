import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from config import DATASET_DIR, MODELS_DIR

# ----------------------------
# Load Dataset
# ----------------------------

csv_file = DATASET_DIR / "landmarks.csv"

df = pd.read_csv(csv_file)

print(df.head())

# ----------------------------
# Split Features & Labels
# ----------------------------

X = df.drop("label", axis=1).values
y = df["label"].values

# ----------------------------
# Encode Labels
# ----------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

# Save encoder
with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# ----------------------------
# Train/Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

# ----------------------------
# Neural Network
# ----------------------------

model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(63,)))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))

model.add(Dense(len(label_encoder.classes_), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ----------------------------
# Training
# ----------------------------

early_stop = EarlyStopping(
    patience=15,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
)

# ----------------------------
# Evaluation
# ----------------------------

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy :", accuracy)

predictions = model.predict(X_test)

predictions = np.argmax(predictions, axis=1)

print(classification_report(
    y_test,
    predictions,
    target_names=label_encoder.classes_
))

# ----------------------------
# Save Model
# ----------------------------

model.save(MODELS_DIR / "gesture_model.keras")

print("\nModel Saved!")

# ----------------------------
# Plot Accuracy
# ----------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train")

plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()