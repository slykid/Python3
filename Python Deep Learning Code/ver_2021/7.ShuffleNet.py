import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, DepthwiseConv2D, Conv2D, BatchNormalization, Activation, Permute, Reshape, AvgPool2D, Add, Concatenate, Input, MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.models import Model

def channel_shuffle(x, groups):
    _, width, height, channels = x.get_shape().as_list()

    group_channels = channels // groups

    x = Reshape([width, height, group_channels, groups])(x)
    x = Permute([1, 2, 4, 3])(x)
    x = Reshape([width, height, channels])(x)

    return x


def unit(x, groups, channels, strides):
    y = x

    x = Conv2D(channels//4, kernel_size=1, strides=1, padding="same", groups=groups)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = channel_shuffle(x, groups)

    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding="same")(x)
    x = BatchNormalization()(x)

    if strides == 2:
        channels = channels - y.shape[-1]

    x = Conv2D(channels, kernel_size=1, strides=1, padding="same", groups=groups)(x)

    if strides == 1:
        x = Add()([x, y])
    elif strides == 2:
        y = AvgPool2D(pool_size=3, strides=2, padding="same")(y)
        x = Concatenate([x, y])

    x = Activation('relu')(x)

    return x

def shufflenet_v1(n_classes, start_channels, input_shape=(224,224,3)):
    groups = 2
    input = Input(input_shape)

    x = Conv2D(kernel_size=3, strides=2, padding="same", use_bias=True)(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    repetitions = [3, 7, 3]

    for i, repetition in enumerate(repetitions):
        channels = start_channels * (2 ** i)
        x = unit(x, groups, channels, strides=2)

        for i in range(repetition):
            x = unit(x, groups, channels, strides=1)

    x = GlobalAvgPool2D()(x)
    output = Dense(n_classes, activation="softmax")(x)

    model = Model(input, output)

    return model