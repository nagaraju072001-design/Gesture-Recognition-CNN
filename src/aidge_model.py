import pickle
import numpy as np

import aidge_core
import aidge_backend_cpu
import aidge_onnx

from config import MODELS_DIR


class AidgeGestureModel:
    """
    Aidge-based gesture classifier.

    Input:
        63 normalized hand-landmark values
        (21 landmarks × 3 coordinates)

    Output:
        Gesture label and confidence.
    """

    def __init__(self):
        print("Loading Aidge AI model...")

        # Load ONNX model
        self.model = aidge_onnx.load_onnx(
            str(MODELS_DIR / "gesture_model.onnx")
        )

        # Compile for Raspberry Pi CPU
        self.model.compile(
            "cpu",
            aidge_core.dtype.float32,
            dims=[[1, 63]]
        )

        # Load the original label encoder
        with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        # Create Aidge inference scheduler
        self.scheduler = aidge_core.SequentialScheduler(self.model)

        print("✅ Aidge AI model loaded successfully.")

    def predict(self, landmarks):
        """
        Predict gesture from 63 normalized landmarks.
        """

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        ).reshape(1, 63)

        # Convert NumPy input to Aidge Tensor
        input_tensor = aidge_core.Tensor(landmarks)

        # Run inference
        outputs = self.scheduler.forward(
            forward_dims=False,
            data=[input_tensor]
        )

        # Convert Aidge output to NumPy
        probabilities = np.asarray(outputs[0])

        # Find predicted class
        class_id = int(np.argmax(probabilities[0]))

        # Get confidence
        confidence = float(probabilities[0][class_id])

        # Decode class using original label encoder
        gesture = self.label_encoder.inverse_transform(
            [class_id]
        )[0]

        return gesture, confidence
