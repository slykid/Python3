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

