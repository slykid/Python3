# 1. Conv2D Layers


# 2. Conv2D with Filter
# 1) Shape with Filters
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

N, n_H, n_W, n_C = 1, 28, 28, 3
n_filter = 5  # 필터 수
k_size = 3    # 필터 크기

images = tf.random.uniform(minval=0, maxval=1, shape=(N, n_H, n_W, n_C))

conv = Conv2D(filters=n_filter, kernel_size=k_size)
Y = conv(images)
W, b = conv.get_weights()

print("Input images: {}".format(images.shape))
print("W/B: {} / {}".format(W.shape, b.shape))
print("Output images: {}".format(Y.shape))

# 2) Computations with Filters
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D

N, n_H, n_W, n_C = 1, 28, 28, 3
n_filter = 5  # 필터 수
k_size = 3    # 필터 크기

images = tf.random.uniform(minval=0, maxval=1, shape=(N, n_H, n_W, n_C))

conv = Conv2D(filters=n_filter, kernel_size=k_size)
Y = conv(images)

print(Y.shape)
print("Y(Tensorflow) \n", Y.numpy())
print(Y.numpy().squeeze().shape)                    # squeeze() : 불필요한 차원을 제거함
print(Y.numpy().squeeze().swapaxes(0, -1).shape)    # swapaxes(): 차원 변경
print(np.transpose(Y.numpy().squeeze(), (2, 0, 1)))

Y = np.transpose(Y.numpy().squeeze(), (2, 0, 1))
print("Y(Tensorflow) \n", Y)

################################################################
# 참고자료. 차원 변경
import numpy as np

images = np.random.randint(low=0, high=10, size=(2, 3, 4))

print(images.shape)
print(images)

tmp = np.transpose(images, (2, 0, 1))
print(tmp.shape)
################################################################
## forward propagation (manual)
W, b = conv.get_weights()
images = images.numpy().squeeze()

print(W.shape, b.shape)

Y_man = np.zeros(shape=(n_H - k_size + 1, n_W - k_size + 1, n_filter))
for c in range(n_filter):
    c_W = W[:, :, :, c]
    c_b = b[c]

    print(c_W.shape, c_b.shape)

    for h in range(n_H - k_size + 1):
        for w in range(n_W - k_size + 1):
            window = images[h:h + k_size, w:w + k_size, :]
            conv = np.sum(window * c_W) + c_b

            Y_man[h, w, c] = conv

print(np.transpose(Y_man, (2, 0, 1)))

# 3. Model Implementation with Conv2D Layer
## 1) Models with Sequential Method
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

n_neurons = [10, 20, 30]

model = Sequential()
model.add(Conv2D(filters=n_neurons[0], kernel_size = 3, activation='relu'))
model.add(Conv2D(filters=n_neurons[1], kernel_size = 3, activation='relu'))
model.add(Conv2D(filters=n_neurons[2], kernel_size = 3, activation='relu'))

x = tf.random.normal(shape=(32, 28, 28, 3))
pred = model(x)

print("Input: {}".format(x.shape))
print("Output: {}".format(pred.shape))

for layer in model.layers:
    W, b = layer.get_weights()
    print(W.shape, b.shape)
print("====================================")
trainable_variables = model.trainable_variables
for train_var in trainable_variables:
    print(train_var.shape)

# 2) Models with Model sub-classing
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

n_neurons = [10, 20, 30]

class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()
        global n_neurons

        self.conv_layers = []
        for n_neuron in n_neurons:
            self.conv_layers.append(Conv2D(filters=n_neuron, kernel_size=3, activation="relu"))

    def call(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x

model = TestModel()
x = tf.random.normal(shape=(32, 28, 28, 3))
pred = model(x)

print("Input: {}".format(x.shape))
print("Output: {}".format(pred.shape))

for layer in model.layers:
    W, b = layer.get_weights()
    print(W.shape, b.shape)
print("====================================")
trainable_variables = model.trainable_variables
for train_var in trainable_variables:
    print(train_var.shape)


