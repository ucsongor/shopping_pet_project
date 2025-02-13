import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

# Loading Mnist dataset for fitting
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalisation

# Model creation
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # 10 számjegy (0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fitting the model
history = model.fit(x_train[..., np.newaxis], y_train, epochs=5, validation_data=(x_test[..., np.newaxis], y_test))

# Evaulate the model
test_loss, test_acc = model.evaluate(x_test[..., np.newaxis], y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Prediction test
predictions = model.predict(x_test[..., np.newaxis])


# Visualisation
def plot_image(index):
    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"Valós: {y_test[index]}, Predikció: {np.argmax(predictions[index])}")
    plt.axis("off")
    plt.show()


plot_image(0)