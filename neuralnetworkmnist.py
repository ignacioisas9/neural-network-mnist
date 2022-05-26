import os
from pickletools import optimize
import cv2 #Computer Vision.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Cargamos la base de datos MNIST descargadas directamente de tensorflow (tf).
mnist = tf.keras.datasets.mnist

#x_train será la componente que contiene la imagen del digito manuscrito.
#y_train será la componente que contiene a la clasificación de dicha imagen, en este caso, el número que representa al digito manuscrito.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizar los pixeles.
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Modelamos la grilla de 28x28 pixeles.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu')) #relu = rectified linear unit.
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) #10 outputs: 0,1,2,...,9. Y softmax se asegura que la suma de la activacion de los 10 digitos sea 1.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Este numero es probablemente un {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1