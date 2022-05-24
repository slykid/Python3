# 작성자: SLYKID (김 길 현)
# kaggle site: https://www.kaggle.com/competitions/dog-breed-identification

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


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
images = []
classes = []

for i in range(len(label["id"])):
    image = cv2.imread(label["path"][i])
    images.append(cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)))
    classes.append(species[species["species"]==label["breed"][i]].index[0])

# 학습 데이터 생성
features = np.zeros((len(classes), 224, 224, 3), dtype="float32")
labels = keras.utils.to_categorical(classes, len(species["index"]))

for i in range(len(classes)):
    img = keras.preprocessing.image.load_img(label["path"][i], target_size=(224, 224))  # TODO: 내용 수정
    img = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(img.copy(), axis=0)
    features[i] = x / 255.0

x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, stratify=labels)

# 모델링
input_layer = keras.Input(shape=(224, 224, 3))
cnn1_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(input_layer)
cnn1_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(cnn1_1)
max_pool1 = layers.MaxPool2D(pool_size=(5, 5))(cnn1_2)

cnn2_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(max_pool1)
cnn2_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(cnn2_1)
max_pool2 = layers.MaxPool2D(pool_size=(2, 2))(cnn2_2)

flatten = layers.Flatten()
