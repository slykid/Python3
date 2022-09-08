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
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt

print(tf.__version__)  # 2.9.1

# variables
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# make image path
train = pd.read_csv("data/google_landmark/train.csv")
print(train.info)
print(train.dtypes)
len(pd.unique(train["landmark_id"]))  # 81313

# 분류 문제이므로 class 를 문자열로 변환함
train["landmark_id"] = train["landmark_id"].apply(lambda x: str(x))

train_path = glob.glob("data\\google_landmark\\train\\*\\*\\*\\*.jpg")
df_path = pd.DataFrame(train_path, columns=["path"])
df_path["id"] = df_path["path"].apply(lambda x: x.split("\\")[-1].split(".jpg")[0])
train_prep = train.merge(df_path, how="inner", on="id", sort=True)
train_prep = train_prep[["id", "path", "landmark_id"]]
print(train_prep.dtypes)

# 학습 데이터 셋 생셩
## 학습:검증 = 8:2
list_ds = tf.data.Dataset.list_files(train_prep["path"], shuffle=False)

valid_size = int(len(train_prep["path"]) * 0.2)
train_ds = list_ds.skip(valid_size)
valid_ds = list_ds.take(valid_size)

def get_label(file_path):
    label = train_prep.loc[file_path == train_prep["path"], "landmark_id"][0]
    return tf.argmax(label)

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    return img, label

# 데이터셋 생성
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# Check
for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

# Modeling

