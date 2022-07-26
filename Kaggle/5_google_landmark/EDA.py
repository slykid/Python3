import os
import glob
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, InputLayer, DepthwiseConv2D, ReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from matplotlib import pyplot as plt

print(tf.__version__)  # 2.8.0

# variables
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# make image path
label = pd.read_csv("data/google_landmark/train.csv")
label["file_path"] = label["id"].apply(lambda x: "data/google_landmark/train/" + x[0] + "/" + x[1] + "/" + x[2] + "/" + x + ".jpg")

train_files = label["file_path"].to_list()

# load sample image
sample_img = cv2.imread(np.random.choice(train_files))
plt.imshow(sample_img)
print("image shape:" + sample_img.shape + "\n")

# image processing
data_generator = ImageDataGenerator(
    rescale=1./255., validation_split=0.2, rotation_range=20,
    zoom_range=0.1, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.1, horizontal_flip=True, fill_mode="nearest"
)

train_generator = data_generator.flow_from_dataframe(
    label,
    directory='data/google_landmark/train/',
    x_col='file_path', y_col='landmark_id',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode='categorical', batch_size=BATCH_SIZE,
    shuffle=True, seed=1234, subset='training'
)


# Model
def custom_cnn(inputs, _filters, _kernel_size, _strides):
    x = Conv2D(filters=_filters, kernel_size=_kernel_size, strides=_strides, padding="same")(inputs)
    x = BatchNormalization()(x)

    return ReLU()(x)

def bottleneck(inputs, _filters, _kernels, t, _strides, r=False):
    '''
    :param inputs: Tensor, input tensor of conv. layer
    :param _filters: Integer, dimension of output
    :param _kernels: tuple, (width, height)
    :param t: Integer, expansion factor
    :param _strides: tuple, stride size
    :param r: boolean, Use residual
    :return: Tensor
    '''

    t_channel = input.shape[-1] * t

    x = custom_cnn(inputs, t_channel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel_size=_kernels, strides=_strides, depth_multiplier=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=_filters, kernel_size=(1, 1), strides=_strides, padding="same")(x)
    x = BatchNormalization()(x)

    if r:
        x = tf.keras.layers.add([x, inputs])

    return x

def inverse_residual_block(inputs, _filters, _kernels, _t, _strides, n):
    x = bottleneck(inputs, _filters=_filters, _kernels=_kernels, t=_t, _strides=_strides)

    for i in range(1, n):
        x = bottleneck(inputs, _filters=_filters, _kernels=_kernels, t=_t, 1, True)

    return x

def custom_mobilenet(input_shape, k, plot_model=False):
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = custom_cnn(inputs, 32, (3, 3), _strides=(2, 2))

    x = inverse_residual_block(x, 16, (3, 3), _t=1, _strides=(1, 1), n=1)
    x = inverse_residual_block(x, 24, (3, 3), _t=6, _strides=(2, 2), n=2)
    x = inverse_residual_block(x, 32, (3, 3), _t=6, _strides=(2, 2), n=3)
    x = inverse_residual_block(x, 64, (3, 3), _t=6, _strides=(2, 2), n=4)
    x = inverse_residual_block(x, 96, (3, 3), _t=6, _strides=(1, 1), n=3)
    x = inverse_residual_block(x, 160, (3, 3), _t=6, _strides=(2, 2), n=3)
    x = inverse_residual_block(x, 320, (3, 3), _t=6, _strides=(1, 1), n=1)

    x = custom_cnn(x, 1280, (1, 1), _strides=(1, 1))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 1280))(x)
    x = tf.keras.layers.Dropout(0.3, name="Dropout")(x)
    x = tf.keras.layers.Conv2D(kernel_size=k, (1, 1), padding="same")(x)
    x = tf.keras.layer.Activation("softmax", name="final_activation")(x)

    output = tf.keras.layers.Reshape((k,), name="output")(x)

    model = tf.keras.Model(inputs, output)
    model.summary()

    if plot_model:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shape=True)

    return model

model = custom_mobilenet((224, 224, ), 64, False)
optimizer = Adam(learning_rate=0.05)

model.compile()

