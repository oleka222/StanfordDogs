# -*- coding: utf-8 -*-
"""
Created on Sat May 15 12:26:51 2021

@author: aleks
"""
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

import tensorflow.keras.layers as tfl

batch_size = 32
img_size = (160, 160)
directory = "images/Images"
train_dataset = image_dataset_from_directory(directory, shuffle=True, batch_size=batch_size, image_size=img_size, validation_split=0.2, subset='training', seed=42)
validation_dataset = image_dataset_from_directory(directory,shuffle=True, batch_size=batch_size, image_size=img_size, validation_split=0.2, subset='validation', seed=42)


class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        
        
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

input_shape = img_size + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

base_model.trainable = False
inputs = tf.keras.Input(shape=input_shape) 

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tfl.GlobalAveragePooling2D()(x)
x = tfl.Dropout(0.2)(x)
prediction_layer = tfl.Dense(120, activation = "softmax")

outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.005
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
initial_epochs = 5
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

import numpy as np


img = tf.keras.preprocessing.image.load_img("kiler.jpg", target_size=(160, 160))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
predictions = model.predict(img)
print(class_names[np.argmax(predictions)])

img = tf.keras.preprocessing.image.load_img("dante.jpg", target_size=(160, 160))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
predictions2 = model.predict(img)
print(class_names[np.argmax(predictions2)])


img = tf.keras.preprocessing.image.load_img("dante2.jpg", target_size=(160, 160))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
predictions3 = model.predict(img)
print(class_names[np.argmax(predictions3)])

img = tf.keras.preprocessing.image.load_img("dante3.jpg", target_size=(160, 160))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
predictions4 = model.predict(img)
print(class_names[np.argmax(predictions4)])

img = tf.keras.preprocessing.image.load_img("dante4.jpg", target_size=(160, 160))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
predictions5 = model.predict(img)
print(class_names[np.argmax(predictions5)])

