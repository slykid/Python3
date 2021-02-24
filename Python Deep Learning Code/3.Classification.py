import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

## 데이터 로드
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")

print(red.head())
print(white.head())

## 신규 컬럼 추가
red['type'] = 0
white['type'] = 1

wine = pd.concat([red, white])
print(wine.describe)

plt.hist(wine['type'])
plt.xticks([0, 1])
plt.show()

print(wine['type'].value_counts())
print(wine.info())

## 정규화
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
print(wine_norm.head())
print(wine_norm.describe())

## 셔플링
wine_shuffle = wine_norm.sample(frac=1)  # sample(): 매개변수인 frac 의 값만큼 전체 데이터에서 추출한 후 랜덤하게 shuffle 함
print(wine_shuffle.head())

## numpy array로 변환
wine_array = wine_shuffle.to_numpy()
print(wine_array[:5])

## train - test split
train_idx = int(len(wine_array) * 0.8)
x_train, y_train = wine_array[:train_idx, :-1], wine_array[:train_idx, -1]
x_test, y_test = wine_array[train_idx:, :-1], wine_array[train_idx:, -1]

## One-Hot Encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

print(y_train[0])
print(y_test[0])

## 모델링
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.07),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

## 학습하기
history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.25)

## 성능 확인
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_acc')
plt.xlabel('Epochs')
plt.ylim(0.5, 1)
plt.legend()

plt.show()

## 결과 예측하기
### 1) predict() 메소드로 결과에 대한 확률치 계산
y_pred = model.predict(x_test)
print(y_pred[0])

### 2) around() 함수로 결과 처리
prediction = np.around(y_pred, 1)
print(prediction)

### 3) argmax() 함수로 최종 결과 산출
result = np.argmax(prediction, axis=1)
print(result)

### 위의 과정을 한번에 수행하는 경우
y_pred_final = np.argmax(model.predict(x_test), axis=1)
print(y_pred_final)