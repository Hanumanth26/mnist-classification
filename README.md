# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

![Screenshot 2024-03-18 144117](https://github.com/Hanumanth26/mnist-classification/assets/121033192/e5c7d23f-e976-4b79-81e7-d86711bf9127)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing.


### STEP 2:
Build a CNN model.

### STEP 3:

Compile and fit the model and then predict.
## PROGRAM

### Name:HANUMANTH A
### Register Number:212222240016

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import keras as kf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[59999]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add (layers. Input (shape=(28,28,1)))
model.add (layers. Conv2D (filters=32, kernel_size=(5,5), activation='relu'))
model.add (layers. MaxPool2D (pool_size=(2,2)))
model.add (layers. Flatten())
model.add (layers. Dense (32, activation='relu'))
model.add (layers. Dense (16, activation='relu'))
model.add (layers. Dense (8, activation='relu'))
model.add (layers. Dense (10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train_scaled,y_train_onehot,epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled, y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(classification_report(y_test,x_test_predictions))
img = image.load_img('deep.png')
type(img)
img = image.load_img('deep.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-03-18 144158](https://github.com/Hanumanth26/mnist-classification/assets/121033192/7bc2de21-a225-4bc3-a5c8-53aa47530b13)

![Screenshot 2024-03-18 144214](https://github.com/Hanumanth26/mnist-classification/assets/121033192/837a0bf5-93b0-4310-a9ab-78b4ce48b51f)


### Classification Report


![Screenshot 2024-03-18 144225](https://github.com/Hanumanth26/mnist-classification/assets/121033192/ef426ec7-eff0-41d2-bca0-ca6fd0e2c041)

### Confusion Matrix

![Screenshot 2024-03-18 144249](https://github.com/Hanumanth26/mnist-classification/assets/121033192/c34b90b8-06a3-4310-9280-2a4efd292ef2)
![Screenshot 2024-03-18 144236](https://github.com/Hanumanth26/mnist-classification/assets/121033192/599cab79-96da-4135-910d-d76b47af39f1)


### New Sample Data Prediction

![Screenshot 2024-03-18 144249](https://github.com/Hanumanth26/mnist-classification/assets/121033192/60c95173-5d59-4776-aa01-fcc9d7b7da71)


## RESULT
Include your result here.
