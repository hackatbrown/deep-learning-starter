import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    """Builds a simple feedforward neural network for MNIST classification."""
    model = models.Sequential([
        # TODO: Flatten the 28x28 image input
        # TODO: Add first Dense layer (e.g., 128 neurons, ReLU)
        # TODO: Add second Dense layer (e.g., 64 neurons, ReLU)
        # TODO: Add output layer (10 classes, softmax)
    ])

    model.compile(
        optimizer='___',  # e.g. 'adam'
        loss='___',       # e.g. 'sparse_categorical_crossentropy'
        metrics=['___']   # e.g. 'accuracy'
    )
    return model
