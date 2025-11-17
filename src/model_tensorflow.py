"""
This module builds a small TensorFlow/Keras neural network for MNIST.
It sets up the layers, compiles the model with loss and optimizer settings,
and exposes a function the training script can call to create the model.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    """Builds a simple feedforward neural network for MNIST classification."""
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model