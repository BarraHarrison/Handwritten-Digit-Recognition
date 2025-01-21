# Handwritten Digit Recognition in Python
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import os

# mnist = tf.keras.datasets.mnist
# # X being the pixel data (image) and Y being the classification (number)
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()

# # adding layers to the model
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(10, activation="softmax"))

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# # Train the model
# model.fit(x_train, y_train, epochs=3)
# model.save("Handwritten.keras")

model = tf.keras.models.load_model("Handwritten.keras")

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
    except:
        pass