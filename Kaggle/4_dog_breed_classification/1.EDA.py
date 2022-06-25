# 작성자: SLYKID (김 길 현)
# kaggle site: https://www.kaggle.com/competitions/dog-breed-identification
import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import glob
import PIL
from PIL import Image

import tensorflow as tf
from tensorflow import keras

label = pd.read_csv("data/dog-breed-identification/labels.csv")

# 이미지 경로 생성
label.info()
label["path"] = label["id"].map(lambda x: "data/dog-breed-identification/train/" + x + ".jpg")
label = label[["id", "path", "breed"]]

# 견종 확인 및 라벨링
print(len(pd.unique(label["breed"])))  # 120 종
species = pd.DataFrame(pd.unique(label["breed"]), columns=["species"]).reset_index()

# 이미지 정제
IMG_WIDTH = 250
IMG_HEIGHT = 250
BATCH_SIZE = 32
images = []
classes = []

## 이미지 로드




# 모델링
input = keras.Input(shape=(224, 224, 3), batch_size=64, name="input_layer")
conv1 = tf.keras.layers.Conv2D(filter=32, kernel_size=3, strides=3,  padding='same')(input)
pool1 = tf.keras.layers.MaxPool2D()(conv1)
norm1 = tf.keras.layers.BatchNormalization()(pool1)

conv2 = tf.keras.layers.Conv2D(filter=32, kernel_size=3, strides=3, padding="same")(norm1)
pool2 = tf.keras.layers.MaxPool2D()(conv2)
norm2 = tf.keras.layers.BatchNormalization()(pool2)

flatten = tf.keras.layers.Flatten()(norm2)
dense1 = tf.keras.layers.Dense(256)(flatten)
model = tf.keras.