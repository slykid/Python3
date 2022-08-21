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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt

print(tf.__version__)  # 2.8.0

# variables
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# make image path
train = pd.read_csv("data/google_landmark/train.csv")

train_list = glob.glob("data/google_landmark/train/*/*/*/*.jpg")
for i in range(0, len(train_list)):
    train_list[i] = train_list[i].replace("\\", "/", 5)

# load sample image
sample_img = cv2.imread(np.random.choice(train_files))
plt.imshow(sample_img)
print("image shape:" + str(sample_img.shape) + "\n")

# image processing
data_generator = ImageDataGenerator(
    rescale=1./255., validation_split=0.2, rotation_range=20,
    zoom_range=0.1, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.1, horizontal_flip=True, fill_mode="nearest"
)

train_generator = data_generator.flow_from_dataframe(
    label,
    directory='data/google_landmark/train',
    x_col='file_path', y_col='landmark_id',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode='categorical', batch_size=BATCH_SIZE,
    shuffle=True, seed=1234, subset='training'
)

# test dataset
test_list = glob.glob("data/google_landmark/test/*/*/*/*.jpg")
for i in range(0, len(test_list)):
    test_list[i] = test_list[i].replace("\\", "/", 5)

test_label = pd.DataFrame(test_list, columns=["path"])
test_label["id"] = test_label["path"].apply(lambda x: x.split("/")[-1].split(".jpg")[0])
test_label = test_label[["id", "path"]]

test_generator = ImageDataGenerator(rescale=1./255.)

test_generator = test_generator.flow_from_dataframe(
    test_label,
    directory='data/google_landmark/test',
    x_col="filename", target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode=None, batch_size=BATCH_SIZE, shuffle=False
)


# index dataset
index_list = glob.glob("data/google_landmark/index/*/*/*/*.jpg")
for i in range(0, len(index_list)):
    index_list[i] = index_list[i].replace("\\", "/", 5)

index_label = pd.DataFrame(index_list, columns=["path"])
index_label["id"] = index_label["path"].apply(lambda x: x.split("/")[-1].split(".jpg")[0])
index_label = index_label[["id", "path"]]



# Model
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

history = model.fit(train_generator, epochs=50, callbacks=[early_stop])
