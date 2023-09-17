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
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.2)

model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

image_folder = 0
image_number = 1
coincidences = 0
total_images = 50

for image_folder in range(10):
    print(f"Número: {image_folder}")
    while os.path.isfile(f"digits/{image_folder}/digit{image_number}.png"):
        try:
            img = cv2.imread(f"digits/{image_folder}/digit{image_number}.png")[:,:,0]
            img = np.array([img])
            prediction = model.predict(img)
            print(f"Este numero es probablemente un {np.argmax(prediction)}")
            # plt.imshow(img[0], cmap=plt.cm.binary)
            # plt.show()
            if np.argmax(prediction) == image_folder:
                coincidences += 1
                precision = coincidences/total_images
        except:
            print("Error!")
        finally:
            image_number += 1
    image_number = 1
print(f"precision: {precision}")