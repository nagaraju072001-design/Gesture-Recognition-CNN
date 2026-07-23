import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
)


def build_model(num_classes):

    model = Sequential()

    # Input Layer
    model.add(Dense(
        256,
        activation="relu",
        input_shape=(63,),
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    # Hidden Layer 1
    model.add(Dense(
        128,
        activation="relu",
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    # Hidden Layer 2
    model.add(Dense(
        64,
        activation="relu",
    ))
    model.add(BatchNormalization())

    # Hidden Layer 3
    model.add(Dense(
        32,
        activation="relu",
    ))

    # Output Layer
    model.add(Dense(
        num_classes,
        activation="softmax",
    ))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
