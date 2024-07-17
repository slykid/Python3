import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.math import exp
from tensorflow.linalg import matmul

# 1.1 Shape of Dense Layer
N, n_features = 8, 10
n_neurons = 3
x = tf.random.normal(shape=(N, n_features))

dense = Dense(units=n_neurons, activation='sigmoid')
y_pred = dense(x)

W, B = dense.get_weights()
print("===== Input Weights/Bias =====")
print("X: ", x.shape)
print("W: ", W.shape)
print("B: ", B.shape)
print("y_pred: ", y_pred.shape)

# Output Calculations
N, n_features = 4, 10
n_neurons = 3
X = tf.random.normal(shape=(N, n_features))

dense = Dense(units=n_neurons, activation='sigmoid')
y_pred = dense(X)

W, B = dense.get_weights()
print("y_pred(Tensorflow): \n", y_pred.numpy())

Z = matmul(X, W) + B
y_manual = 1/(1 + exp(-Z))
print("y_manual(Manual): \n", y_manual.numpy())

# calculation dot products
y_manual_vector = np.zeros(shape=(N, n_neurons))
print(y_manual_vector)

for x_idx in range(N):
    x = X[x_idx]

    for nu_idx in range(n_neurons):
        w, b = W[:, nu_idx], B[nu_idx]

        z = tf.reduce_sum(x * w) + b
        a = 1 / (1 + np.exp(-z))
        y_manual_vector[x_idx, nu_idx] = a

print("Y with dot products: \n", y_manual_vector)


# 1.2 Cascaded Dense Layers
N, n_features = 4, 10
n_neurons = [3, 5]

X = tf.random.normal(shape=(N, n_features))

dense1 = Dense(units=n_neurons[0], activation='sigmoid')
dense2 = Dense(units=n_neurons[1], activation='sigmoid')

# forward propagation
A1 = dense1(X)
Y = dense2(A1)

# get Weights/Bias
W1, B1 = dense1.get_weights()
W2, B2 = dense2.get_weights()

print("X: {}\n".format(X.shape))

print("W1: {}".format(W1.shape))
print("B1: {}".format(B1.shape))
print("A1: {}\n".format(A1.shape))

print("W2: {}".format(W2.shape))
print("B2: {}".format(B2.shape))
print("Y: {}\n".format(Y.shape))

# Dense Layers with Python List(Layer 옵션값 설정 꿀팁!)
N, n_features = 4, 10
n_neurons = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

X = tf.random.normal(shape=(N, n_features))

# 반복문 & List 를 활용한 Layer 옵션값 설정하기
dense_layers = list()
for n_neuron in n_neurons:
    dense = Dense(units=n_neuron, activation='relu')
    dense_layers.append(dense)

# 예시: enumerate 를 활용한 Layer 설정 및 설정완료 메세지 출력
print("Input: ", X.shape)
for dense_idx, dense in enumerate(dense_layers):
    X = dense(X)
    print("After dense layer", dense_idx)
    print(X.shape, "\n")
