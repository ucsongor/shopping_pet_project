import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from MNIST_model import MNISTModel
import matplotlib
matplotlib.use('TkAgg')

# Adatok betöltése
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test / 255.0  # Normalizálás

# Modell betöltése
mnist_model = MNISTModel()
mnist_model.load_model()

# Egy tesztkép kiértékelése
index = 0  # Ezt módosíthatod, hogy más képet nézz
predicted_label = mnist_model.predict(x_test[index])

# Kép megjelenítése
plt.imshow(x_test[index], cmap='gray')
plt.title(f"Valós: {y_test[index]}, Predikció: {predicted_label}")
plt.axis("off")
plt.show()
