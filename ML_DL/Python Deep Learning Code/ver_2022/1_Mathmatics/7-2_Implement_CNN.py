import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense

# 1.1 Implementation with Sequential Method
N, n_H, n_W, n_C = 4, 28, 28, 3
n_conv_neuron = [10, 20, 30]
n_dense_neuron = [50, 30, 30]
kernel_size, padding = 3, "same"
pool_size, pool_strides = 2, 2
activation='relu'

x = tf.random.normal(shape=(N, n_H, n_W, n_C))

model = Sequential()
model.add(Conv2D(filters=n_conv_neuron[0], kernel_size=kernel_size, padding=padding, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
model.add(Conv2D(filters=n_conv_neuron[1], kernel_size=kernel_size, padding=padding, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
model.add(Conv2D(filters=n_conv_neuron[2], kernel_size=kernel_size, padding=padding, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
model.add(Flatten())

model.add(Dense(units=n_dense_neuron[0], activation=activation))
model.add(Dense(units=n_dense_neuron[1], activation=activation))
model.add(Dense(units=n_dense_neuron[2], activation='softmax'))

y_pred = model(x)
print(y_pred.shape)

# Using For-loop
model = Sequential()
for conv_neuron in n_conv_neuron:
    model.add(Conv2D(filters=conv_neuron, kernel_size=kernel_size, padding=padding, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
model.add(Flatten())
for dense_neuron in n_dense_neuron:
        model.add(Dense(units=dense_neuron, activation=activation))
model.add(Dense(units=n_dense_neuron[-1], activation='softmax'))

y_pred = model(x)
print(y_pred.shape)


# 1.2 Implementation with Model Subclassing
class TestCNN(Model):
    def __init__(self):
        super(TestCNN, self).__init__()

        self.conv1 = Conv2D(filters=n_conv_neuron[0], kernel_size=kernel_size, padding=padding, activation='relu')
        self.conv1_pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)

        self.conv2 = Conv2D(filters=n_conv_neuron[1], kernel_size=kernel_size, padding=padding, activation='relu')
        self.conv2_pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)

        self.conv3 = Conv2D(filters=n_conv_neuron[2], kernel_size=kernel_size, padding=padding, activation='relu')
        self.conv3_pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)

        self.flatten = Flatten()

        self.dense1 = Dense(units=n_dense_neuron[0], activation=activation)
        self.dense2 = Dense(units=n_dense_neuron[1], activation=activation)
        self.dense3 = Dense(units=n_dense_neuron[2], activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        print("Conv1: ", x.shape)
        x = self.conv1_pool(x)
        print("Pool1: ", x.shape)

        x = self.conv2(x)
        print("Conv2: ", x.shape)
        x = self.conv2_pool(x)
        print("Pool2: ", x.shape)

        x = self.conv3(x)
        print("Conv3: ", x.shape)
        x = self.conv3_pool(x)
        print("Pool3: ", x.shape)

        x = self.flatten(x)
        print("Flatten: ", x.shape)

        x = self.dense1(x)
        print("Dense1: ", x.shape)

        x = self.dense2(x)
        print("Dense2: ", x.shape)

        x = self.dense3(x)
        print("Dense3: ", x.shape)

        return x

model = TestCNN()
y_pred = model(x)

# 1.3 Model Implementation using Sequential + Layer Sub-classing
class MyConv(Layer):
    def __init__(self, n_neuron):
        super(MyConv, self).__init__()

        self.conv = Conv2D(filters=n_neuron, kernel_size=kernel_size, padding=padding, activation='relu')
        self.pool = MaxPooling2D(pool_size=pool_size, strides=pool_strides)

    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)

        return x

model = Sequential()
model.add(MyConv(n_conv_neuron[0]))
model.add(MyConv(n_conv_neuron[1]))
model.add(MyConv(n_conv_neuron[2]))

model.add(Flatten())

model.add(Dense(units=n_dense_neuron[0], activation=activation))
model.add(Dense(units=n_dense_neuron[1], activation=activation))
model.add(Dense(units=n_dense_neuron[2], activation='softmax'))


# 1.4 Model Implementation using Model & Layer Sub-classing
class MyCNN(Model):
    def __init__(self):
        super(MyCNN, self).__init__()

        self.conv1 = MyConv(n_conv_neuron[0])
        self.conv2 = MyConv(n_conv_neuron[1])
        self.conv3 = MyConv(n_conv_neuron[2])

        self.flatten = Flatten()

        self.dense1 = Dense(units=n_dense_neuron[0], activation=activation)
        self.dense2 = Dense(units=n_dense_neuron[1], activation=activation)
        self.dense3 = Dense(units=n_dense_neuron[2], activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        print("Conv1: ", x.shape)

        x = self.conv2(x)
        print("Conv2: ", x.shape)

        x = self.conv3(x)
        print("Conv3: ", x.shape)

        x = self.flatten(x)
        print("Flatten: ", x.shape)

        x = self.dense1(x)
        print("Dense1: ", x.shape)

        x = self.dense2(x)
        print("Dense2: ", x.shape)

        x = self.dense3(x)
        print("Dense3: ", x.shape)

        return x