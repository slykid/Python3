# 1. Shapes in CNN
# 1) Shapes in the Feature Extractors
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

N, n_H, n_W, n_c = 32, 28, 28, 3
n_conv_filter = 5
batch_size = 32
k_size = 3
pool_size = 2
pool_strides = 2

input = tf.random.normal(shape=(N, n_H, n_W, n_c))

conv1 = Conv2D(filters=n_conv_filter, kernel_size=k_size, padding="same", activation="relu")
pool1 = MaxPooling2D(pool_size=pool_size, strides=pool_strides)

conv2 = Conv2D(filters=n_conv_filter, kernel_size=k_size, padding="same", activation="relu")
pool2 = MaxPooling2D(pool_size=pool_size, strides=pool_strides)

flatten = Flatten()

x = conv1(input)
W, b = conv1.get_weights()
print("After conv1: {}".format(x.shape))
print("W/B: {}/{}".format(W.shape, b.shape))

x = pool1(x)
print("After pool1: {}".format(x.shape))

x = conv2(x)
W, b = conv2.get_weights()
print("After conv2: {}".format(x.shape))
print("W/B: {}/{}".format(W.shape, b.shape))

x = pool2(x)
print("After pool2: {}".format(x.shape))

x = flatten(x)
print("After flatten: {}".format(x.shape))

# 2) Shapes in the Classifier
n_neurons = [50, 25, 10]

dense1 = Dense(units=n_neurons[0], activation='relu')
dense2 = Dense(units=n_neurons[1], activation='relu')
dense3 = Dense(units=n_neurons[2], activation='relu')

print("Input Feature: {}".format(x.shape))
x = dense1(x)
W, b = dense1.get_weights()
print("After conv1: {}".format(x.shape))
print("W/B: {}/{}".format(W.shape, b.shape))

x = dense2(x)
W, b = dense2.get_weights()
print("After conv1: {}".format(x.shape))
print("W/B: {}/{}".format(W.shape, b.shape))

x = dense3(x)
W, b = dense3.get_weights()
print("After conv1: {}".format(x.shape))
print("W/B: {}/{}".format(W.shape, b.shape))
