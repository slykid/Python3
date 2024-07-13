import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Constant

# 1.1 Affine Function
x = tf.constant([[10.]])  # input setting

dense = Dense(units=1, activation="linear")  # imp. an affine function

tmp = dense.get_weights()
print(tmp)
# []
# 공백인 이유는 위에 선언된 Dense Layer에 별도의 초기값을 넣어 주지 않았기 때문에, 초기화 되지 않은 상태임
# 실제로 해당 Layer 의 가중치 및 편향이 초기화 되는 시점은 y_pred 를 계산하는 시점임
del tmp

y_pred = dense(x)  # forward propagation + params initialization
print(y_pred)

W, B = dense.get_weights()  # get weights & Bias
print(W, B)

y_manual = tf.linalg.matmul(x, W) + B  # forward propagation(manual)

# print results
print("===== Input/Weight/Bias =====")
print("x: {}\n{}\n".format(x.shape, x.numpy()))
print("W: {}\n{}\n".format(W.shape, W))
print("B: {}\n{}\n".format(B.shape, B))

print("===== Outputs =====")
print("y(Tensorflow): {}\n{}\n".format(y_pred.shape, y_pred.numpy()))
print("y(Manual): {}\n{}\n".format(y_manual.shape, y_manual.numpy()))
# ===== Input/Weight/Bias =====
# x: (1, 1)
# [[10.]]

# W: (1, 1)
# [[-1.4223826]]

# B: (1,)
# [0.]

# ===== Outputs =====
# y(Tensorflow): (1, 1)
# [[-14.223825]]

# y(Manual): (1, 1)
# [[-14.223825]]


# 1.2 Params Initialization
x = tf.constant([[10.]])

w, b = tf.constant(10.), tf.constant(20.)
w_init, b_init = Constant(w), Constant(b)  # 위의 constant 와 다름

print(w, b)  # tf.Tensor(10.0, shape=(), dtype=float32) tf.Tensor(20.0, shape=(), dtype=float32)
print(w_init, b_init)  # Object 타입으로 단순히 값을 갖고 있는 것이 아님

# w_init, b_init 을 이용해 초기화 하는 방법
dense = Dense(units=1, activation='linear', kernel_initializer=w_init, bias_initializer=b_init)

y_pred = dense(x)
W, B = dense.get_weights()

# print results
print(y_pred)
print("W: {}\n{}\n".format(W.shape, W))
print("B: {}\n{}\n".format(B.shape, B))


# 1.3 Affine Function with n Features
x = tf.random.uniform(shape=(1, 10))
print(x.shape, "\n", x)

dense = Dense(units=1)

y_pred = dense(x)

W, B = dense.get_weights()

y_manual = tf.linalg.matmul(x, W) + B

# print results
print("===== Input/Weight/Bias =====")
print("x: {}\n{}\n".format(x.shape, x.numpy()))
print("W: {}\n{}\n".format(W.shape, W))
print("B: {}\n{}\n".format(B.shape, B))

print("===== Outputs =====")
print("y(Tensorflow): {}\n{}\n".format(y_pred.shape, y_pred.numpy()))
print("y(Manual): {}\n{}\n".format(y_manual.shape, y_manual.numpy()))