# 1. 저수준 API 방식
import numpy as np
import random
from matplotlib import pyplot as plt

import tensorflow as tf

x = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

a = tf.Variable(random.random())
b = tf.Variable(random.random())
c = tf.Variable(random.random())

def compute_loss():
    y_pred = a * x * x + b * x + c
    loss = tf.reduce_mean((y - y_pred)**2)

    return loss

optimizer = tf.keras.optimizers.Adam(lr=0.07)
for i in range(1000):
    # 잔차제곱의 평균을 최소화시킴
    optimizer.minimize(compute_loss, var_list=[a,b,c])

    if i % 100 == 99:
        print(i+1, 'a : ', a.numpy(), ' ,b: ', b.numpy(), ' ,c: ', c.numpy(), ' ,loss: ', compute_loss().numpy())

line_x = np.arange(min(x), max(x), 0.01)
line_y = a * line_x * line_x + b * line_x + c

plt.plot(line_x, line_y, 'r--')
plt.plot(x, y, 'bo')
plt.xlabel('Population Growth Rate (%)')
plt.ylabel('Eldery Population Rate (%)')


# 2. 고수준 API 방식
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=6, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
model.summary()

model.fit(x, y, epochs=10)
model.predict(x)

line_x = np.arange(min(x), max(x), 0.01)
line_y = model.predict(line_x)

plt.plot(line_x, line_y, 'r--')
plt.plot(x, y, 'bo')
plt.xlabel('Population Growth Rate (%)')
plt.ylabel('Eldery Population Rate (%)')

# 실습 : 보스턴 주택 가격 예측하기
import tensorflow as tf
import pandas as pd
import tensorflow.feature_column as fc
from tensorflow.keras.datasets import boston_housing

# 데이터 로드
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Pandas 데이터 프레임으로 변환
features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x_train_df = pd.DataFrame(x_train, columns=features)
x_test_df = pd.DataFrame(x_test, columns=features)
y_train_df = pd.DataFrame(y_train, columns=["MEDV"])
y_test_df = pd.DataFrame(y_test, columns=["MEDV"])

# 추정기를 위한 입력함수 생성
def estimator_input(df_data, df_label, epochs=10, shuffle=True, batch_size=32):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(df_data), df_label))
        if shuffle:
            dataset.shuffle(100)

        dataset = dataset.batch(batch_size).repeat(epochs)

        return dataset

    return input_function

# 특징 열 추출 & 파이프라인 생성
feature_columns = []
for feature in features:
    feature_columns.append(fc.numeric_column(feature, dtype=tf.float32))

train_input = estimator_input(x_train_df, y_train_df)
valid_input = estimator_input(x_test_df, y_test_df, epochs=1, shuffle=False)

# LinearRegressor 추정기 생성
linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
linear_est.train(train_input)
result = linear_est.evaluate(valid_input)

# 예측하기
y_pred = linear_est.predict(valid_input)

for pred, exp in zip(y_pred, y_test[:32]):
    print("Predicted value: ", pred['predictions'][0], " Expected: ", exp)