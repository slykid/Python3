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
import tensorflow as tf

N, n_features = 100, 5
batch_size = 32

t_weights = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
t_bias = tf.constant([10], dtype=tf.float32)

X = tf.random.normal(mean=0, stddev=1, shape=(N, n_features))
Y = tf.reduce_sum(t_weights * X, axis=1) + t_bias

# for batch_idx in range(N // batch_size):
#     x = X[batch_idx * batch_size : (batch_idx + 1) * batch_size]
#     y = Y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
# 위의 과정대로 생성할 수 있지만 매우 복잡함

# 이를 위해 아래와 같이 텐서플로에서 제공하는 함수를 사용하면 쉬움
# 또한 데이터를 어떻게 뽑아서 모델에 주는 지에 대한 것도 중요함!
dataset = tf.data.Dataset.from_tensor_slices((X, Y))  # from_tensor_slices 는 사용자가 임의로 텐서를 생성하는 경우에만 사용함
dataset = dataset.batch(batch_size)

for x, y in dataset:
    print("X: {} \n\nY: {}".format(x, y))
    print(x.shape, y.shape)

# 2. Loss Functions
# 2-1. MSE (Mean Squared Error)
# 2-1-1. Calculation
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

N, n_features = 100, 5
batch_size = 32

loss_object = MeanSquaredError()

batch_size = 32
predictions = tf.random.normal(shape=(batch_size, 1))
labels = tf.random.normal(shape=(batch_size, 1))

mse = loss_object(labels, predictions)
mse_manual = tf.reduce_mean(tf.math.pow(labels - predictions, 2))
print("MSE(Tensorflow): ", mse.numpy())  # MSE(Tensorflow):  2.1063626
print("MSE(manual): ", mse_manual.numpy())  # MSE(manual):  2.1063626

# 2-1-2. MSE with Model/Dataset
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError

N, n_features = 100, 5
batch_size = 32

X = tf.random.normal(shape=(N, n_features))
Y = tf.random.normal(shape=(N, 1))

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(batch_size)

model = Dense(units=1, activation="linear")
loss_object = MeanSquaredError()

for x, y in dataset:
    predictions = model(x)
    loss = loss_object(y, predictions)

    print("Loss: " + str(loss.numpy()))


# 2-2. Binary Cross Entropy
# 2-2-1. BCE Calculation
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

batch_size = 32
n_classes = 2

predictions = tf.random.uniform(shape=(batch_size, 1))
labels = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=n_classes, dtype=tf.int32)

print(predictions)
print(labels)

loss_object = BinaryCrossentropy()
loss = loss_object(labels, predictions)

labels = tf.cast(labels, tf.float32)
bce_manual = -(labels * tf.math.log(predictions) + (1 - labels) * tf.math.log(1 - predictions))
bce_manual = tf.reduce_mean(bce_manual)

print("BCE(Tensorflow): ", loss.numpy())  # BCE(Tensorflow):  0.8595582
print("BCE(Manual): ", bce_manual.numpy())  # BCE(Manual):  0.85955864

# 2-2-2. BCE with Model/Dataset
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

N, n_features = 8, 5

t_weights = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
t_bias = tf.constant([10], dtype=tf.float32)

X = tf.random.normal(mean=0, stddev=1, shape=(N, n_features))
Y = tf.reduce_sum(t_weights * X, axis=1) + t_bias
Y = tf.cast(Y > 5, tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(batch_size)

model = Dense(units=1, activation='sigmoid')
loss_object = BinaryCrossentropy()

for x, y in dataset:
    predictions = model(x)
    loss = loss_object(y, predictions)
    print("Loss: ", loss.numpy())  # Loss:  0.68282413

# 2-3. Sparse Categorical Cross Entropy
# 2-3-1. SCCE Calculation
import tensorflow as tf

from tensorflow.keras.losses import SparseCategoricalCrossentropy

batch_size, n_classes = 16, 5

predictions = tf.random.uniform(shape=(batch_size, n_classes), minval=0, maxval=1, dtype=tf.float32)
pred_sum = tf.reshape(tf.reduce_sum(predictions, axis=1), (-1, 1))
print(predictions.shape, pred_sum.shape)

predictions /= pred_sum
labels = tf.random.uniform(shape=(batch_size, ), minval=0, maxval=n_classes, dtype=tf.int32)

print(labels)

loss_object = SparseCategoricalCrossentropy()
loss = loss_object(labels, predictions)
print("Loss: ", loss.numpy())  # Loss:  1.7217846

ce = 0
for label, prediction in zip(labels, predictions):
    ce += -tf.math.log(prediction[label])
ce /= batch_size
print("Cross Entropy: ", ce.numpy())  # Cross Entropy:  1.7217847

# 2-3-2. SCCE with Model/Dataset
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

N, n_features = 100, 2
n_classes = 3
batch_size = 16

X = tf.zeros(shape=(0, n_features))
Y = tf.zeros(shape=(0, 1), dtype=tf.int32)

for class_idx in range(n_classes):
    center = tf.random.uniform(minval=-15, maxval=15, shape=(2, ))

    x1 = center[0] + tf.random.normal(shape=(N, 1))
    x2 = center[1] + tf.random.normal(shape=(N, 1))

    x = tf.concat((x1, x2), axis=1)
    y = class_idx * tf.ones(shape=(N, 1), dtype=tf.int32)

    X = tf.concat((X, x), axis=0)
    Y = tf.concat((Y, y), axis=0)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(batch_size)

model = Dense(units=n_classes, activation='softmax')
loss_object = SparseCategoricalCrossentropy()

for x, y in dataset:
    predictions = model(x)
    loss = loss_object(y, predictions)
    print(loss.numpy())

print("\nLoss: " + str(loss.numpy()))  # Loss: 0.62953514

# 2-4. Categorical Cross Entropy
# 2-4-1. CCE Calculation
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy

batch_size, n_classes = 16, 5

predictions = tf.random.uniform(shape=(batch_size, n_classes), minval=0, maxval=1, dtype=tf.float32)
pred_sum = tf.reshape(tf.reduce_sum(predictions, axis=1), (-1, 1))
print(predictions.shape, pred_sum.shape)

predictions = predictions/pred_sum
labels = tf.random.uniform(shape=(batch_size, ), minval=0, maxval=n_classes, dtype=tf.int32)
labels = tf.one_hot(labels, n_classes)

print(labels)

loss_object = CategoricalCrossentropy()
loss = loss_object(labels, predictions)
print("Loss: ", loss.numpy())  # Loss: 1.8669639

cce_man = tf.reduce_mean(tf.reduce_sum(-labels * tf.math.log(predictions), axis=1))
print("CCE (Manual): ", cce_man.numpy())  # CCE (Manual):  1.8669641

# 2-4-2. CCE with Model/Dataset
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy

N, n_features = 8, 2
n_classes = 5
batch_size = 32

X = tf.zeros(shape=(0, n_features))
Y = tf.zeros(shape=(0, ), dtype=tf.int32)

for class_idx in range(n_classes):
    center = tf.random.uniform(minval=-15, maxval=15, shape=(2, ))

    x1 = center[0] + tf.random.normal(shape=(N, 1))
    x2 = center[1] + tf.random.normal(shape=(N, 1))

    x = tf.concat((x1, x2), axis=1)
    y = class_idx * tf.ones(shape=(N, ), dtype=tf.int32)

    X = tf.concat((X, x), axis=0)
    Y = tf.concat((Y, y), axis=0)

Y = tf.one_hot(Y, depth=n_classes, dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(batch_size)

model = Dense(units=n_classes, activation="softmax")
loss_object = CategoricalCrossentropy()

for x, y in dataset:
    predictions = model(x)
    loss = loss_object(y, predictions)
    print(loss.numpy())

print("Loss: ", loss.numpy())  # Loss:  7.047408
