import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


class MNISTModel:
    def __init__(self, model_path="mnist_model.h5"):
        self.model = None
        self.model_path = model_path
        self._build_model()

    def _build_model(self):
        """ Build the CNN model """
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test, epochs=5):
        """ Fitting the model """
        history = self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        self.save_model()  # Automatikus mentés tanítás után
        return history

    def save_model(self):
        """ Saving the model """
        self.model.save(self.model_path)
        print(f"Model saved: {self.model_path}")

    def load_model(self):
        """ If model exists, then loads the model """
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"Loading the model: {self.model_path}")
            return True
        else:
            print("No existing model yet, please start with making one.")
            return False

    def predict(self, image):
        """ Prediction for a given image """
        image = np.expand_dims(image, axis=0)  # (28, 28) -> (1, 28, 28)
        image = np.expand_dims(image, axis=-1)  # (1, 28, 28) -> (1, 28, 28, 1)
        prediction = self.model.predict(image)
        return np.argmax(prediction)
