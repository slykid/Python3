# 작성일자: 2022.07.04
# 작성자: 김길현
# Competition: https://www.kaggle.com/competitions/dog-breed-identification
# 참고자료: https://www.kaggle.com/code/atrisaxena/using-tensorflow-2-x-to-classify-breeds-of-dogs

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf

# 데이터 로드 및 전처리
## 1. 라벨 데이터 전처리
labels = pd.read_csv("data/dog-breed-identification/labels.csv")
labels["filename"] = labels["id"].apply(lambda x: x + ".jpg")
labels = labels[["id", "filename", "breed"]]

## 2. 경로 리스트 생성
img_common_path = "data/dog-breed-identification/"
filenames = (img_common_path + "train/" + labels["filename"]).tolist()

## 3. 분류 라벨 리스트 생성
classes = labels["breed"].unique()
target = [breed for breed in labels["breed"]]
target_encode = [label == np.array(classes) for label in target]

## 4. 학습, 검증 셋 분리
x_train, x_valid, y_train, y_valid = train_test_split(filenames, target_encode, test_size=0.2, random_state=1234)
print(len(x_train), len(y_train), len(x_valid), len(y_valid))

## 5. 이미지 전처리
sample_image = plt.imread(x_train[0])
plt.imshow(sample_image)

tf.image.convert_image_dtype(sample_image, tf.float64)
