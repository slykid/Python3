import numpy as np
import tensorflow as tf

# 1. Make Dataset
# - 원하는 형식으로 데이터셋을 만들어볼 수 있어야 정확하게 학습이 어떻게 이뤄지는 지 알 수 있음

# 1-1. Dataset for Regression
N, n_features = 8, 5
t_weight = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
t_bias = tf.constant([10], dtype=tf.float32)

print(t_weight)
print(t_bias)

X = tf.random.normal(mean=0, stddev=1, shape=(N, n_features))
print(X.shape, t_weight.shape, t_bias.shape)

Y = t_weight * X + t_bias  # 잘못된 연산(cuz, W x X 한 후 axis = 1 을 기준으로 합한 후 해당 결과에 b 를 더해야함)
print(X.shape, '\n', X)
print(Y.shape, '\n', Y)

Y = tf.reduce_sum(t_weight * X, axis=1) + t_bias
print(X.shape, '\n', X)
print(Y.shape, '\n', Y)

# 1-2. Dataset for Binary Classification
# - Regression 때와 동일하지만 아래의 과정이 추가됨
print(Y)

Y = tf.cast(Y >= 5, tf.int32)
print(Y)

# 1-3. Dataset for Multi Classification
import tensorflow as tf
from matplotlib import pyplot as plt
plt.style.use("seaborn")

tf.random.set_seed(1234)

N, n_features = 30, 2
n_classes = 5

X = tf.zeros(shape=(0, n_features))
Y = tf.zeros(shape=(0, 1), dtype=tf.int32)

fig, ax = plt.subplots(figsize=(5, 5))
for class_index in range(n_classes):
    center = tf.random.uniform(minval=-15, maxval=15, shape=(2, ))
    # ax.scatter(center[0], center[1])

    x1 = center[0] + tf.random.normal(shape=(N, 1))
    x2 = center[1] + tf.random.normal(shape=(N, 1))

    x = tf.concat((x1, x2), axis=1)
    y = class_index * tf.ones(shape=(N, ), dtype=tf.int32)

    ax.scatter(x[:, 0].numpy(), x[:, 1].numpy(), alpha=0.3)

    X = tf.concat((X, x), axis=0)
    Y = tf.concat((Y, y), axis=0)

Y = tf.one_hot(Y, depth=n_classes, dtype=tf.int32)

print("X(shape/dtype/data): {} / {}\n{}\n".format(X.shape, X.dtype, X.numpy()))
print("Y(shape/dtype/data): {} / {}\n{}\n".format(Y.shape, Y.dtype, Y.numpy()))

# 1-4. Dataset Object