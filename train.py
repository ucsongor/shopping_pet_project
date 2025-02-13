import numpy as np
from tensorflow import keras
from MNIST_model import MNISTModel

# Load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation
x_train, x_test = x_train / 255.0, x_test / 255.0

# Expand dimensions for CNN
x_train = np.expand_dims(x_train, axis=-1)  # (60000, 28, 28) -> (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)

# Copying
mnist_model = MNISTModel()

# Load model if exists
if not mnist_model.load_model():
    # Ha nincs mentett modell, tanítsuk be
    mnist_model.train(x_train, y_train, x_test, y_test, epochs=5)
