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
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

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

## 이미지 크기조정 전처리 함수
def prep_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float64)
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    return image

## 이미지-라벨 매핑 함수
def get_image_label(path, label):
    image = prep_image(path)

    return image, label

## 데이터셋 생성함수
def create_dataset(x, y=None, batch_size = 32,_valid=False, _test=False):
    # 검증용 데이터 셋 생성
    if _valid:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        data = data.map(get_image_label).batch(batch_size)

        return data

    # 테스트 데이터 셋 생성
    elif _test:
        data = tf.data.Dataset.from_tensor_slices(tf.constant(x))
        data = data.map(prep_image).batch(batch_size)

        return data

    # 학습용 데이터 셋 생성
    else:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y))).shuffle(buffer_size=len(x))
        data = data.map(get_image_label).batch(batch_size)

        return data

train_set = create_dataset(x_train, y_train)
valid_set = create_dataset(x_valid, y_valid)

# 6. 모델링
base_model = MobileNetV2(include_top=False, classes=len(classes))

base_model.trainable = False

input = tf.keras.layers.Input(shape=(224, 224, 3))
model = base_model(input, training=False)
model = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(model)
model = tf.keras.layers.Dropout(0.2)(model)
output = tf.keras.layers.Dense(len(classes), activation="softmax")(model)

clf_model = tf.keras.Model(input, output)
clf_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

EarlyStoppingCallbacks = tf.keras.callbacks.EarlyStopping(\
    monitor='val_loss', patience=2, baseline=None, restore_best_weights=True\
)

# 모델 학습하기
history = clf_model.fit(\
    train_set,\
    epochs=100,\
    validation_data=valid_set,\
    callbacks=[EarlyStoppingCallbacks]\
)

# 테스트 데이터 생성
test_path = "data/dog-breed-identification/test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]
test_dataset = create_dataset(test_filenames, _test=True)

y_pred = clf_model.predict(test_dataset)

df_preds = pd.DataFrame(columns=["id"] + list(classes))
df_preds["id"] = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
df_preds[list(classes)] = y_pred

df_preds.to_csv("result/dog-breed-identification/submission2.csv", index=False)