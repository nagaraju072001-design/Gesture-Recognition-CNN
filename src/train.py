import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from config import DATASET_DIR, MODELS_DIR
from network import build_model

# -----------------------------------
# Load Dataset
# -----------------------------------

csv_file = DATASET_DIR / "landmarks.csv"

df = pd.read_csv(csv_file)

print(df.head())

# -----------------------------------
# Features & Labels
# -----------------------------------

X = df.drop("label", axis=1).values
y = df["label"].values

# -----------------------------------
# Encode Labels
# -----------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# -----------------------------------
# Train/Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training:", len(X_train))
print("Testing :", len(X_test))

# -----------------------------------
# Build Model
# -----------------------------------

model = build_model(len(label_encoder.classes_))

# -----------------------------------
# Callbacks
# -----------------------------------

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
    ),

    ModelCheckpoint(
        str(MODELS_DIR / "gesture_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),

]

# -----------------------------------
# Train
# -----------------------------------

history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=100,

    batch_size=32,

    callbacks=callbacks,

)

# -----------------------------------
# Evaluate
# -----------------------------------

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

pred = model.predict(X_test)

pred = pred.argmax(axis=1)

print(
    classification_report(
        y_test,
        pred,
        target_names=label_encoder.classes_,
    )
)

# -----------------------------------
# Plot
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Train")

plt.plot(history.history["val_accuracy"], label="Validation")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.legend()

plt.grid(True)

plt.show()
