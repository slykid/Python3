# 작성자: SLYKID (김 길 현)
# kaggle site: https://www.kaggle.com/competitions/dog-breed-identification

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
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
    img = keras.preprocessing.image.load_image(label["path"][i], target_size=(224, 224))  # TODO: 내용 수정
    img = keras.preprocessing.image.image_to_array(img)
    x = np.expand_dims(img.copy(), axis=0)
    features[i] = x / 255.0


x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, stratify=True)

# 모델링
model = keras.model.Sequential([

])