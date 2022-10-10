import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential

# 샘플데이터 생성하기
x = []
y = []

for i in range(6):
    lst = list(range(i, i + 4))

    x.append(list(map(lambda c: [c/10], lst)))
    y.append((i + 4) / 10)

x, y = np.array(x), np.array(y)

for i in range(len(x)):
    print(x[i], y[i])

# 모델링
model = Sequential([
    SimpleRNN(units=10, return_sequences=False, input_shape=[4, 1]),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

model.fit(x, y, epochs=100, verbose=0)
print(model.predict(x))

print(model.predict(np.array([[[0.6], [0.7], [0.8], [0.9]]])))
print(model.predict(np.array([[[-0.1], [0.0], [0.1], [0.2]]])))