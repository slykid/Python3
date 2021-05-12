import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

# AlexNet 모델 생성
model = Sequential()

## 1 계층 (conv1 - pool1 - batch1)
model.add(Conv2D(96, (11, 11), strides=4, input_shape=(224, 224, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(BatchNormalization())

## 2 계층 (conv2 - pool2 - batch2)
model.add(ZeroPadding2D(2))
model.add(Conv2D(256, (5, 5), strides=1, activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(BatchNormalization())

## 3 계층 (conv3 - pool3 - batch3)
model.add(ZeroPadding2D(1))
model.add(Conv2D(384, (3, 3), strides=1, activation="relu"))

## 4 계층 (conv4)
model.add(ZeroPadding2D(1))
model.add(Conv2D(384, (3, 3), strides=1, activation="relu"))

## 5 계층 (conv5 - pool5)
model.add(ZeroPadding2D(1))
model.add(Conv2D(256, (3, 3), strides=1, activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

## 1차원 배열로 Flatten
model.add(Flatten())

## 6 계층 (FC6)
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))

## 7 계층 (FC7)
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))

## 8 계층
model.add(Dense(1, activation='sigmoid'))

## 손실함수 정의
loss_func = SGD(lr=0.01, decay=5e-4, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=loss_func, metrics=["accuracy"])
model.summary()