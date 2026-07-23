import pickle
import numpy as np

from tensorflow.keras.models import load_model

from config import MODELS_DIR


class GestureModel:
    """
    Loads the TensorFlow model and label encoder,
    then provides a simple prediction interface.
    """

    def __init__(self):

        print("Loading AI model...")

        self.model = load_model(MODELS_DIR / "gesture_model.keras")

        with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        print("✅ AI model loaded successfully.")

    def predict(self, landmarks):

        landmarks = np.array(
            landmarks,
            dtype=np.float32
        ).reshape(1, -1)

        prediction = self.model.predict(
            landmarks,
            verbose=0
        )

        class_id = np.argmax(prediction)

        confidence = float(prediction[0][class_id])

        gesture = self.label_encoder.inverse_transform(
            [class_id]
        )[0]

        return gesture, confidence
