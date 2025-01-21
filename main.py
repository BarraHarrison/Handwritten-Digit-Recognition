# Handwritten Digit Recognition in Python
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

mnist = tf.keras.datasets.mnist
# X being the pixel data (image) and Y being the classification (number)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()