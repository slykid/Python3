import os
import glob
import cv2
import numpy as np
import pandas as pd
import shutil

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, InputLayer, DepthwiseConv2D, ReLU
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt

print(tf.__version__)  # 2.9.1

# variables
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# make image path
train = pd.read_csv("data/google_landmark/train.csv")
print(train.info)
print(train.dtypes)
len(pd.unique(train["landmark_id"]))  # 81313

# 분류 문제이므로 class 를 문자열로 변환함
train["landmark_id"] = train["landmark_id"].apply(lambda x: str(x))

train_path = glob.glob("data\\google_landmark\\train\\*\\*\\*\\*.jpg")
df_path = pd.DataFrame(train_path, columns=["path"])
df_path["id"] = df_path["path"].apply(lambda x: x.split("\\")[-1].split(".jpg")[0])
train_prep = train.merge(df_path, how="inner", on="id", sort=True)
train_prep = train_prep[["id", "path", "landmark_id"]]
print(train_prep.dtypes)

# 학습 데이터 셋 생셩
## 학습:검증 = 8:2
list_ds = tf.data.Dataset.list_files(train_prep["path"], shuffle=False)

valid_size = int(len(train_prep["path"]) * 0.2)
train_ds = list_ds.skip(valid_size)
valid_ds = list_ds.take(valid_size)

def get_label(file_path):
    label = train_prep.loc[file_path == train_prep["path"], "landmark_id"][0]
    return tf.argmax(label)

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    return img, label

# 데이터셋 생성
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# Check
for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

# Modeling
def _conv_block(inputs, filters, kernel, strides):
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return ReLU()(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = inputs.shape[-1] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def custom_mobilenet(input_shape, k, plot_model=False):

    inputs = tf.keras.layers.Input(shape=input_shape, name='input')
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 1280))(x)
    x = tf.keras.layers.Dropout(0.3, name='Dropout')(x)
    x = tf.keras.layers.Conv2D(k, (1, 1), padding='same')(x)
    x = tf.keras.layers.Activation('softmax', name='final_activation')(x)
    output = tf.keras.layers.Reshape((k,), name='output')(x)
    model = tf.keras.models.Model(inputs, output)
    model.summary()
    if plot_model:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    return model

model = custom_mobilenet((224, 224, 1), 64, False)
optimizer = Adam(learning_rate=0.05)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

history = model.fit(train_ds, epochs=50, callbacks=[early_stop])
