import numpy as np
import pandas as pd

import tensorflow as tf

# tensorflow 버전 확인
print(tf.__version__)

# etc. Tensorflow-GPU 사용 시, GPU Device 선택하기
from tensorflow.python.client import device_lib
device_lib.list_local_devices()  # 현재 검색되는 GPU 는 1개(0번) 만 검색됨

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 번에 해당하는 GPU 모듈만 사용하게 됨

# 1. Tensorflow Basic
## 1) 난수 생성
## - 균일분포
rand_num = tf.random.uniform([1], 0, 1)
print(rand_num)
print(rand_num.shape)

rand_num = tf.random.uniform([2,2], 0, 1)
print(rand_num)


## - 정규분포
rand_num = tf.random.normal([2], 0, 1)
print(rand_num)


## 2) 뉴런 생성하기
## - 뉴런 생성 1
import numpy as np
import pandas as pd
import math

import tensorflow as tf

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

x = 1
y = 0
w = tf.random.normal([1], 0, 1)

output = sigmoid(x * w)
print(output)

for i in range(1000):
    output = sigmoid(x * w)
    err = y - output
    w = w + x * 0.1 * err

    if i % 100 == 99:
        print(i+1, err, output)


# example. bad gradient descent
x = 0
y = 1
w = tf.random.normal([1], 0, 1)

for i in range(1000):
    output = sigmoid(x * w)
    error = y - output
    w = w + x * 0.1 * error

    if i % 100 == 99:
        print(i+1, error, output)

# example. gradient descent complete
x = 0
y = 1
w = tf.random.normal([1], 0, 1)
b = tf.random.normal([1], 0, 1)

for i in range(1000):
    output = sigmoid(x * w + 1 * b)
    error = y - output
    w = w + x * 0.1 * error

    if i % 100 == 99:
        print(i+1, error, output)

## 3) keras
import tensorflow as tf
from tensorflow import keras

CLASSES = 10
RESHAPE = 784

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(CLASSES, input_shape=(RESHAPE,),
          kernel_initializer='zeros', name='dense_layer', activation='softmax'))
print(model.summary())

# tensorflow-datasets
import tensorflow as tf
import tensorflow_datasets as tfds

builders = tfds.list_builders()
print(builders)

data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data['train'], data['test']

print(info)