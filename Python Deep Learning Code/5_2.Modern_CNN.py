import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

# 1. AlexNet 모델 생성
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


# 2. VGGNet
import tensorflow as tf
from tensorflow import keras as K

classes = 10

## 모델 구성
model = tf.keras.models.Sequential()

## 제 1 계층
model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

## 제 2 계층
model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

## 제 3 계층
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

## 제 4 계층
model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

## 제 5 계층
model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(classes, activation='softmax', name='predictions'))

print(model.summary())

model = K.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=10,
    classifier_activation="softmax"
)

print(model.summary())

## GoogLeNet
## 참고자료
# - https://ichi.pro/ko/tensorflowleul-sayonghan-googlenet-inceptionv1-174269272563335
# - https://poddeeplearning.readthedocs.io/ko/latest/CNN/GoogLeNet/
import tensorflow as tf
from tensorflow import keras as K

# inception module
def inception(x, f_1x1, f_reduce_3x3, f_3x3, f_reduce_5x5, f_5x5, f_pool):
    # 1x1
    path1 = K.layers.Conv2D(f_1x1, strides=1, padding="same", activation="relu")(x)

    # 3x3_reduce -> 3x3
    path2 = K.layers.Conv2D(f_reduce_3x3, strides=1, padding="same", activation="relu")(x)
    path2 = K.layers.Conv2D(f_3x3, strides=1, padding="same", activation="relu")(path2)

    # 5x5_reduce -> 5x5
    path3 = K.layers.Conv2D(f_reduce_5x5, strides=1, padding="same", acitivation="relu")(x)
    path3 = K.layers.Conv2D(f_5x5, strides=1, padding="same", activation="relu")(path3)

    # 3x3_max_pool -> 1x1
    path4 = K.layers.MaxPooling2D(pool_size=(3,3), strides=1, padding="same")(x)
    path4 = K.layers.Conv2D(f_pool, strides=1, padding="same", activation="relu")(path4)

    return tf.concat([path1, path2, path3, path4], axis=3)

# GoogLeNet
input_layer = K.layers.Input(shape=(32, 32, 3))
input = K.layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:])(input_layer)
x = K.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu")(input)
x = K.layers.MaxPooling2D(pool_size=3, strides=2)(x)
x = K.layers.Conv2D(64, 1, strides=1, padding="same", activation="relu")(x)
x = K.layers.MaxPooling2D(pool_size=3, strides=2)(x)

## inception 3a
x = inception(x, f_1x1=64, f_reduce_3x3=96, f_3x3=128, f_reduce_5x5=16, f_5x5=32, f_pool=32)

## inception 3b
x = inception(x, f_1x1=128, f_reduce_3x3=128, f_3x3=192, f_reduce_5x5=32, f_5x5=32, f_pool=32)
