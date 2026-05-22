import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

from config import ACTION_COUNT, STATE_SIZE


def create_model():
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    model = Sequential(
        [
            Input(shape=(STATE_SIZE,)),
            Dense(32, activation="tanh", kernel_initializer=initializer),
            Dense(32, activation="tanh", kernel_initializer=initializer),
            Dense(ACTION_COUNT, activation="softmax", kernel_initializer=initializer),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model=create_model()
model.save("models/upmodel_compact.keras")