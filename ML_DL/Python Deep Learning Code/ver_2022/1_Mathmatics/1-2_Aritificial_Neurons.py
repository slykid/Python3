import tensorflow as tf
from tensorflow.math import exp, maximum
from tensorflow.keras.layers import Dense, Activation

# 1.1 Activation Layers
x = tf.random.normal(shape=(1, 5))  # input setting

# imp. activation functions
sigmoid = Activation('sigmoid')
tanh = Activation('tanh')
relu = Activation('relu')

# forward propagation(Tensorflow)
y_pred_sigmoid = sigmoid(x)
y_pred_tanh = tanh(x)
y_pred_relu = relu(x)

# forward propagation(Manual)
y_manual_sigmoid = 1 / (1 + exp(-x))
y_manual_tanh = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
y_manual_relu = maximum(x, 0)

print("X: {}\n{}".format(x.shape, x.numpy()))
print("Sigmoid(Tensorflow): {}\n{}".format(y_pred_sigmoid.shape, y_pred_sigmoid.numpy()))
print("Sigmoid(Manual): {}\n{}".format(y_manual_sigmoid.shape, y_manual_sigmoid.numpy()))

print("Tanh(Tensorflow): {}\n{}".format(y_pred_tanh.shape, y_pred_tanh.numpy()))
print("Tanh(Manual): {}\n{}".format(y_manual_tanh.shape, y_manual_tanh.numpy()))

print("ReLU(Tensorflow): {}\n{}".format(y_pred_relu.shape, y_pred_relu.numpy()))
print("ReLU(Manual): {}\n{}".format(y_manual_relu.shape, y_manual_relu.numpy()))


# 1.2 Activation in Dense Layer
x = tf.random.normal(shape=(1, 5))  # input setting

# imp. aritificial neurons
dense_sigmoid = Dense(units=1, activation='sigmoid')
dense_tanh = Dense(units=1, activation='tanh')
dense_relu = Dense(units=1, activation='relu')

# forward propagation
y_pred_sigmoid = dense_sigmoid(x)
y_pred_tanh = dense_tanh(x)
y_pred_relu = dense_relu(x)

# print results
print("AN with Sigmoid: {}\n{}".format(y_pred_sigmoid.shape, y_pred_sigmoid.numpy()))
print("AN with Tanh: {}\n{}".format(y_pred_tanh.shape, y_pred_tanh.numpy()))
print("AN with ReLU: {}\n{}".format(y_pred_relu.shape, y_pred_relu.numpy()))

# Proof
print("\n=====\n")
W, B = dense_sigmoid.get_weights()
z = tf.linalg.matmul(x, W) + B
a = 1 / (1 + exp(-z))
print("Activation Value(Tensorflow): {}\n{}".format(y_pred_sigmoid.shape, y_pred_sigmoid.numpy()))
print("Activation Value(Manual): {}\n{}".format(a.shape, a.numpy()))


# 1.3 Review
activation = 'sigmoid'
# activation = 'tanh'
# activation = 'relu'

x = tf.random.uniform(shape=(1, 10))

dense = Dense(units=1, activation=activation)

y_pred = dense(x)
W, B = dense.get_weights()

y_manual = tf.linalg.matmul(x, W) + B
if activation == 'sigmoid':
    y_manual = 1/(1 + exp(-y_manual))
elif activation == 'tanh':
    y_manual = (exp(y_manual) - exp(-y_manual)) / (exp(y_manual) + exp(-y_manual))
elif activation == 'relu':
    y_manual = maximum(y_manual, 0)

print("Activation: ", activation)
print("y_pred: {}\n{}\n".format(y_pred.shape, y_pred.numpy()))
print("y_manual: {}\n{}\n".format(y_manual.shape, y_manual.numpy()))


# 1.4 Mini batches
N, n_features = 8, 10  # set input params
x = tf.random.normal(shape=(N, n_features))  # generate minibatch

dense = Dense(units=1, activation='relu')  # imp. an AN
y_pred = dense(x)  # forward propagation(Tensorflow)

W, B = dense.get_weights()  # get Weights/Bias

# print input/weight/bias
print("Shape of x: ", x.shape)
print("Shape of W: ", W.shape)
print("Shape of B: ", B.shape)

y_manual = tf.linalg.matmul(x, W) + B  # forward propagation(Manual)
y_manual = maximum(y_manual, 0)  # activation ReLU(Manual)

# print results
print("Output(Tensorflow): \n", y_pred.numpy())
print("Output(Manual): \n", y_manual.numpy())