import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization, Dense, AveragePooling2D, Activation
from tensorflow.keras import datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def mobilnet_v1(x, alpha=1):
    def depthwise(x, _padding, _filter, _stride):
        x = DepthwiseConv2D(kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=_filter, kernel_size=1, strides=_stride, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    x = Conv2D(filters=int(32 * alpha), kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = depthwise(x, "same", int(64 * alpha), 1)
    x = depthwise(x, "valid", int(128 * alpha), 2)
    x = depthwise(x, "same", int(128 * alpha), 1)
    x = depthwise(x, "same", int(256 * alpha), 2)
    x = depthwise(x, "same", int(256 * alpha), 1)
    x = depthwise(x, "valid", int(512 * alpha), 2)

    for i in range(5):
        x = depthwise(x, "same", int(512 * alpha), 1)

    x = depthwise(x, "valid", int(1024 * alpha), 2)
    x = depthwise(x, "same", int(1024 * alpha), 1)

    return x

filename = "checkpoint-epochs-{}-batch-{}-trial-001.h5".format(30,128)
checkpoint = ModelCheckpoint(filename, monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
early_stop = EarlyStopping(monitor="val_loss", patience=10)
reduceLR = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)




