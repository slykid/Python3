# 작성자: SLYKID (김 길 현)
# kaggle site: https://www.kaggle.com/competitions/dog-breed-identification

import os
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

label = pd.read_csv("data/dog-breed-identification/labels.csv")

IMG_WIDTH = 224
IMG_HEIGHT = 224
CHANNELS = 3
BATCH_SIZE = 32

# 이미지 경로 생성
label.info()
label["filename"] = label["id"].map(lambda x: x + ".jpg")
label = label[["id", "breed", "filename"]]

label['breed'].value_counts().plot.bar(figsize=(16, 8))

# 학습 데이터 생성
data_generator = ImageDataGenerator(rescale=1./255., validation_split=0.2, rotation_range=20,
                                    zoom_range=0.1, width_shift_range=0.2, height_shift_range=0.2,
                                    shear_range=0.1, horizontal_flip=True, fill_mode="nearest"
                                )

train_generator = data_generator.flow_from_dataframe(
                    label,
                    # directory='../input/dog-breed-identification/train/',  # for KAGGLE Notebook
                    directory='data/dog-breed-identification/train/',
                    x_col='filename', y_col='breed',
                    target_size=(IMG_WIDTH, IMG_HEIGHT), class_mode='categorical', batch_size=BATCH_SIZE,
                    shuffle=True, seed=1234, subset='training'
                )
val_generator = data_generator.flow_from_dataframe(
                    label,
                    # directory='../input/dog-breed-identification/train/',  # for KAGGLE Notebook
                    directory='data/dog-breed-identification/train/',
                    x_col='filename', y_col='breed',
                    target_size=(IMG_WIDTH, IMG_HEIGHT), class_mode='categorical', batch_size=BATCH_SIZE,
                    shuffle=True, seed=1234, subset='validation'
                )

# 제너레이터 이미지 확인
img, label = next(train_generator)
fig = plt.figure(figsize=(15, 10))

for i in range(12):
    fig.add_subplot(3, 4, i+1)
    plt.imshow(img[i])
    plt.axis('off')

# 테스트 데이터 전처리
# test_images = os.listdir('../input/dog-breed-identification/test/') # for KAGGLE Notebook
test_images = os.listdir('data/dog-breed-identification/test/')
test_set = pd.DataFrame(test_images, columns=['filename'])
test_set["id"] = test_set["filename"].apply(lambda x: x.split(".")[0])
test_set = test_set[["id", "filename"]]

# 모델링
base_model = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)))

# 기존 모델 계층은 "학습 안함" 설정
for layer in base_model.layers:
    layer.trainable = False

avg_pool_tail = AveragePooling2D(pool_size=4)(base_model.output)
flatten = Flatten()(avg_pool_tail)
dense1_tail = Dense(1024, activation="relu")(flatten)
dropout1_tail = Dropout(0.3)(dense1_tail)
dense2_tail = Dense(512, activation="relu")(dropout1_tail)
dropout2_tail = Dropout(0.3)(dense2_tail)
result = Dense(120, activation="softmax")(dropout2_tail)

model = tf.keras.Model(inputs=base_model.input, outputs=result)

# Optimizer 설정
optimizer = SGD(learning_rate=0.1, momentum=0.9, decay=0.01)

# Early Stopping 설정
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)

# Checkpoint 설정
# checkpoint = ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True)  # For KAGGLE Notebook
checkpoint = ModelCheckpoint(filepath='data/dog-breed-identification/weight/weights.hdf5', verbose=1, save_best_only=True)  # For KAGGLE Notebook

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

history1 = model.fit(train_generator, epochs=5, validation_data=val_generator, callbacks=[checkpoint])  # loss: 1.1821 - accuracy: 0.7032 - val_loss: 1.2266 - val_accuracy: 0.6996
history2 = model.fit(train_generator, epochs=5, validation_data=val_generator, callbacks=[checkpoint])  # loss: 4.9315 - accuracy: 0.0076 - val_loss: 4.8423 - val_accuracy: 0.0127
history3 = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[checkpoint])
