# 1. tensorflow 사용법
import numpy as np
import pandas as pd
import tensorflow as tf

rand = tf.random.uniform([1], 0, 1)
print(rand)

rand = tf.random.uniform([4], 0, 1)
print(rand)

rand = tf.random.normal([4], 0, 1)
print(rand)

# 신경 구조
import math

## 활성화함수(시그모이드)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

x = 1 ; y = 0
w = tf.random.normal([1], 0, 1)
output = sigmoid(x * w)
print(output)

for i in range(1000):
    output = sigmoid(x)
    error = y - output

    w = w + x * 0.1 * error

    if i % 100 == 99:
        print(i + 1 , error, output)

# bias 적용한 신경
x = 0
y = 1
w = tf.random.normal([1], 0, 1)
b = tf.random.normal([1], 0, 1)

for i in range(1000):
    output = sigmoid(x * w + 1 * b)
    error = y - output
    w = w + x * 0.1 * error
    b = b + 1 * 0.1 * error

    if i % 100 == 99:
        print(i + 1, error, output)

## 신경망 연산 : AND, OR, XOR
print(int(True))  # 1
print(int(False))  # 0

### AND 연산
x = np.array([[1,1], [1,0], [0,1], [0,0]])
y = np.array([[1], [0], [0], [0]])
w = tf.random.normal([2], 0, 1)
b = tf.random.normal([1], 0, 1)
b_x = 1

for i in range(2000):
    error_sum = 0

    for j in range(4):
        output = sigmoid(np.sum(x[j] * w) + b_x * b)
        error = y[j][0] - output
        w = w + x[j] * 0.1 * error
        b = b + b_x * 0.1 * error
        error_sum += error

    if i % 200 == 199:
        print(i + 1, error_sum)

for i in range(4):
    print('X : ', x[i], ', Y : ', y[i], ', Output : ', sigmoid(np.sum(x[i] * w) + b))

### OR 연산
x = np.array([[1,1], [1,0], [0,1], [0,0]])
y = np.array([[1], [1], [1], [0]])
w = tf.random.normal([2], 0, 1)
b = tf.random.normal([1], 0, 1)
b_x = 1

for i in range(2000):
    error_sum = 0

    for j in range(4):
        output = sigmoid(np.sum(x[j] * w) + b_x * b)
        error = y[j][0] - output
        w = w + x[j] * 0.1 * error
        b = b + b_x * 0.1 * error
        error_sum += error

    if i % 200 == 199:
        print(i + 1, error_sum)

for i in range(4):
    print('X : ', x[i], ', Y : ', y[i], ', Output : ', sigmoid(np.sum(x[i] * w) + b))

### XOR 연산
x = np.array([[1,1], [1,0], [0,1], [0,0]])
y = np.array([[0], [1], [1], [0]])
w = tf.random.normal([2], 0, 1)
b = tf.random.normal([1], 0, 1)
b_x = 1

for i in range(2000):
    error_sum = 0

    for j in range(4):
        output = sigmoid(np.sum(x[j] * w) + b_x * b)
        error = y[j][0] - output
        w = w + x[j] * 0.1 * error
        b = b + b_x * 0.1 * error
        error_sum += error

    if i % 200 == 199:
        print(i + 1, error_sum)
print(b)
for i in range(4):
    print('X : ', x[i], ', Y : ', y[i], ', Output : ', sigmoid(np.sum(x[i] * w) + b))

## 2-XOR Network
x = np.array([[1,1], [1,0], [0,1], [0,0]])
y = np.array([[0], [1], [1], [0]])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation="sigmoid", input_shape=(2,)),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
model.summary()

history = model.fit(x, y, epochs=2000, batch_size=1)
model.predict(x)

for w in model.weights:
    print(w)


## 1) Tensor 생성
# numpy 에서의 array
a = np.array([1, 2, 3])
a.shape

### tensorflow의 tensor
a_prime = tf.constant(a)
print(a_prime)
print(a_prime.shape)  # 차원 확인
print(a_prime.dtype)  # 데이터 타입 확인(지정되지 않으면 값에 맞게 임의로 지정함)

print(tf.constant(a, dtype=tf.float32))  # 사용자 지정 형변환
print(a.astype(np.int8))  # numpy 에서 배열의 형변환
print(tf.cast(a_prime, dtype=tf.uint8))  # tensorflow 2.0 에서 텐서의 형변환

a_prime.numpy()  # tensor -> numpy 로 변환방법1
np.array(a_prime)  # tensor -> numpy 로 변환방법2
type(a_prime.numpy())  # Out[25]: numpy.ndarray

### 난수 생성
np.random.randn(9)  # numpy 에서 정규분포를 갖는 난수 생성방법
tf.random.normal([3, 3])  # 정규 분포를 갖는 난수 생성함(normal distribution) / numpy 와 달리 shape를 넣어 줘야 함
tf.random.uniform([3, 3])  # 정규 분포와 관계없이 난수를 생성함(uniform distribution) / numpy 와 달리 shape를 넣어 줘야 함

## 2) 데이터 불러오기
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import datasets  # 텐서플로우에서 제공하고 있는 데이터셋 라이브러리

### mnist 데이터 로드
mnist = datasets.mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x.shape  # 데이터 확인

image = train_x[0]
image.shape  # Out[9]: (28, 28) 입력 뒤에 3이 없다는 의미는 grayscale 의 이미지라는 뜻
plt.imshow(image, 'gray')  # 이미지 출력

train_x = np.expand_dims(train_x, -1)
train_x.shape

new_train_x = tf.expand_dims(train_x, -1)
new_train_x.shape

reshaped = train_x.reshape([60000, 28, 28, 1])
reshaped.shape

new_train_x = train_x[..., tf.newaxis]
new_train_x.shape

new_train_x[0]
plt.imshow(new_train_x[0, :, :, 0 ], "gray")
plt.imshow(np.squeeze(new_train_x[0]), "gray")

train_y.shape
train_y[0]

train_x[0].shape

plt.title(train_y[0])
plt.imshow(train_x[0, :, :, 0], "gray")
plt.show()

## 3) One-hot Encoding
## 컴퓨터가 이해할 수 있는 형태로 변환해 레이블을 부여하는 방식

### Label : 5
label = [0,0,0,0,0,1,0,0,0,0]

### keras 에서 변환함수를 제공함
from tensorflow.keras.utils import to_categorical
to_categorical(5, 10)  #  array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=float32)

label = train_y[0]
label_onehot = to_categorical(label, num_classes=10)
label_onehot

## 4) Layer 별 역할 확인
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

import numpy as np
import pandas as pd

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
image = train_x[0]
image.shape

plt.imshow(image, 'gray')

### 이미지 변환
### - 현재 1개의 이미지만 사용할 것이며, grayscale 로 이미지를 수정할 계획이기 때문에
###   이미지의 shape 값을 (1, 28, 28, 1) 로 변환해주기 위해 아래의 코드를 실행한다.
image = image[tf.newaxis, ..., tf.newaxis]
image.shape

### Feature Extraction
#### (1) Convolution Layer
tf.keras.layers.Conv2D(filters=3,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       padding="SAME",
                       activation='relu')
# tf.keras.layers.Conv2D(3, 3, 1, 'SAME') # 위의 내용과 같은 표현

#### 옵션 살펴보기
# - filters : layer에 대한 output 을 몇 개의 필터로 만들지를 설정
# - kernel_size : filter 의 크기
# - strides : 몇 개의 pixel 을 스킵하면서 읽어나갈 지
# - padding : 이미지의 외곽에 경계를 만들어 줄 것 인지
# - activation : 활성화 함수는 어떤 것을 사용할 지 (반드시 입력할 필요는 없음)

image.dtype  # float32 가 아니므로 강제형변환을 해줘야함(현재는 uint8)
image = tf.cast(image, dtype=tf.float32)
image.dtype

layer = tf.keras.layers.Conv2D(3, 3, 1, padding='SAME')
layer

output = layer(image)
output

plt.subplot(1,2,1)
plt.imshow(image[0, :, :, 0], 'gray')
plt.subplot(1,2,2)
plt.imshow(output[0, :, :, 0], 'gray')
plt.show()

np.min(image), np.max(image)
np.min(output), np.max(output)

#### weight 불러오기
weight = layer.get_weights()  # List 형식임 / [0] 은 weight, [1] 은 bias 임
print(weight)
print(weight[0].shape)
print(weight[1].shape)

#### 시각화 확인
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.hist(output.numpy().ravel(), range=[-2, 2])
plt.ylim(0, 100)
plt.subplot(132)
plt.title(weight[0].shape)
plt.imshow(weight[0][:, :, 0, 0], 'gray')
plt.subplot(133)
plt.title(output.shape)
plt.imshow(output[0, :, :, 0], 'gray')
plt.colorbar()
plt.show()

#### 활성화 함수
tf.keras.layers.ReLU()

activate = tf.keras.layers.ReLU()
act_output = activate(output)

np.min(act_output), np.max(act_output)

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.hist(act_output.numpy().ravel(), range=[-2, 2])
plt.ylim(0, 100)
plt.subplot(122)
plt.title(act_output.shape)
plt.imshow(act_output[0, :, :, 0], 'gray')
plt.show()

#### (2) Pooling
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

import numpy as np
import pandas as pd

pool_layer = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="SAME")
pool_output = pool_layer(act_output)
act_output.shape
pool_output.shape

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.hist(pool_output.numpy().ravel(), range=[-2, 2])
plt.ylim(0, 100)
plt.subplot(122)
plt.title(pool_output.shape)
plt.imshow(pool_output[0, :, :, 0], 'gray')
plt.colorbar()
plt.show()

#### Full connected
##### flatten
import tensorflow as tf
from matplotlib import pyplot as plt

layer = tf.keras.layers.Flatten()
flatten = layer(output)

output.shape
flatten.shape  # 28 * 28 * 3 = 2352

plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.hist(flatten.numpy().ravel())
plt.subplot(212)
plt.imshow(flatten[:, :100], 'jet')
plt.show()

##### Dense
layer = tf.keras.layers.Dense(32, activation="relu")
output = layer(flatten)
output.shape

layer_2 = tf.keras.layers.Dense(10, activation="relu")
output_example = layer_2(output)
output_example.shape

##### Dropout
###### 모델이 오버피팅되는 것을 방지하기 위해서 사용
###### 학습 시에만 dropout을 해주고 실제 evaluation 에서는 모든 노드를 dense 하게 연결하여 사용
layer = tf.keras.layers.Dropout(0.7)
output = layer(output)
output.shape

#### 모델 생성하기
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets

# load datasets
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

# make model
inputs = layers.Input((28, 28, 1))
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(10)(net)  # num_classes
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')

model.summary()  # 배치 사이즈를 지정하지 않았기 때문에 None 으로 표시됨

# loss function
# loss = 'binary_crossentropy'
# loss = 'categorical_crossentropy'

# tf.keras.losses.sparse_categorical_crossentropy  # one hot encoding 이 안된 경우 ex) [1, 3, 5] 와 같이 사용가능
# tf.keras.losses.categorical_crossentropy  # one hot encoding 이 된 경우
# tf.keras.losses.binary_crossentropy  # 클래스가 이진인 경우

loss_func = tf.losses.sparse_categorical_crossentropy

# Metrics
metrics = [tf.keras.metrics.sparse_categorical_accuracy]  # recall, precision 등 여러개를 주기 위해 list 형식으로 선언

# Optimizer
opt = tf.keras.optimizers.Adam()

# Compile
model.compile(optimizer=opt,
              loss=loss_func,
              metrics=metrics)

# check data shape
train_x.shape, train_y.shape
test_x.shape, test_y.shape

# increasse dimension
# gray scale 인 경우에 차원수를 증가시켜줘야함
# tf.expand_dims(train_x, -1).shape  # 기존에 사용하던 방법
train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]

train_x.shape

# rescaling
np.min(train_x), np.max(train_x)

train_x  = train_x / 255.
test_x  = test_x / 255.

np.min(train_x), np.max(train_x)

# training
# set hyperparameter
num_epochs = 1  # 학습 횟수
batch_size = 32  # 메모리를 효율적으로 사용하기 위해서 설정

# fit model
hist = model.fit(train_x, train_y,
          batch_size=batch_size,
          shuffle=True,
          epochs=num_epochs)

print(hist)

## 최적화
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# model
input_shape = (28, 28, 1)
num_classes = 10

# tf.keras.backend.set_floatx('float64')

inputs = layers.Input(input_shape, dtype=tf.float64)
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.5)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.5)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes)(net)  # num_classes
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')

# load datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

x_train, x_test = x_train/255.0, x_test/255.0

# expert 과정(model compile)
# tf.data 활용
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(32)  # test 데이터의 경우 shuffle 은 불필요함 (예측을 맞추기 위함이므로)

# visualize data
for image, label in train_ds.take(2):
    label = np.array(label,dtype=np.uint8)
    plt.title(label[0])
    plt.imshow(image[0, :, :, 0], 'gray')
    plt.show()

image, label = next(iter(train_ds))
image.shape, label.shape

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_ds, epochs = 10000)  # fit 할 때 train_ds 와 같이 데이터 셋을 사용하면 이미지 데이터, 레이블, 배치사이즈 까지 설정되었기 때문에 따로 설정할 필요 없음

# expert 과정(loss function & optimizer)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')    # 손실의 평균을 저장함
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')    # 손실의 평균을 저장함
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#2.0 에서는 세션이 없음
@tf.function  # 세션여는것 없이 바로 사용가능
def train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = model(images)
        loss = loss_object(labels, preds)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, preds)
    
@tf.function
def test_step(images, labels):
    preds = model(images)
    t_loss = loss_object(labels, preds)

    test_loss(t_loss)
    test_accuracy(labels, preds)

for epoch in range(2):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100
                          ))

# model evaluation
num_epochs = 1
batch_size = 64

hist = model.fit(train_x, train_y, batch_size=batch_size, shuffle=True)
hist.history
model.evaluate(test_x, test_y, batch_size=batch_size)  # Loss, Accuracy 순으로 출력

import matplotlib.pyplot as plt
import numpy as np

test_image = test_x[0, :, :, 0]
test_image.shape

plt.title(test_y[0])
plt.imshow(test_image, 'gray')
plt.show()

test_image.shape
pred = model.predict(test_image.reshape(1, 28, 28, 1))
pred.shape
np.argmax(pred)


# test batch로
test_batch = test_x[:32]
test_batch.shape

preds = model.predict(test_batch)
preds.shape

np.argmax(preds, -1)  # 32개에 대한 결과를 확인할 때 사용

for i in range(0, len(np.argmax(preds, -1))):
    plt.imshow(test_batch[i, :, :, 0], 'gray')
    plt.show()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# 2. torch 사용법
## torch 설치 확인
import torch

torch.cuda.get_device_name(0)
torch.cuda.is_available()
print(torch.__version__)



import numpy as np
import torch

# torch 버전 확인
torch.__version__

# 리스트 생성
nums = torch.arange(9)
print(nums)

# 타입확인
type(nums)

# numpy 화
nums.numpy()

# reshape 메소드
nums.reshape(3, 3)

# Random
randoms = torch.rand((3, 3))
print(randoms)

# Zeros
zeros = torch.zeros((3, 3))
print(zeros)

# Ones
ones = torch.ones((3, 3))
print(ones)

# Zero_like
torch.zeros_like(ones)
zeros.size()

# Operations
nums * 3
nums = nums.reshape((3, 3))
nums + nums

result = torch.add(nums, 3)
result.numpy()

# View
range_nums = torch.arange(9).reshape((3, 3))
print(range_nums)

range_nums.view(-1)  # reshape 와 크게 차이 있는 것은 아님

# Slice & Index
print(nums)
print(nums[1])
print(nums[1, 1])
print(nums[1:, 1:])
print(nums[1:])

# Compile
# numpy -> torch
arr = np.array([1, 1, 1])
arr_torch = torch.from_numpy(arr)
print(arr_torch)
arr_torch.float()

# Device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # GPU 및 CUDA 사용 가능환경이면 cuda로 잡힘
print(device)

# AutoGrad
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(out)

out.backward()  # x 에 대해 역전파를 수행
x.grad  # x 에 대한 기울기 확인

print(x.requires_grad)  # True
print((x**2).requires_grad)  # True

with torch.no_grad():  # 기울기 계산 X
    print((x ** 2).requires_grad)  # False

# Data Loader
import torch
from torchvision import datasets, transforms

batch_size = 32
test_batch_size = 32

# data loader를 이용한 데이터 로드
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5,), std=(0.5,))
                   ])),
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=test_batch_size,
    shuffle=True)

images, labels = next(iter(train_loader))
images.shape  # batch size, channel, height, width 순서임 (TF 의 경우 batch size, height, width, channel 순임)
labels.shape

import numpy as np
import matplotlib.pyplot as plt

images[0].shape
torch_image = torch.squeeze(images[0])
torch_image.shape

image = torch_image.numpy()
image.shape

label = labels[0].numpy()
label

plt.title(label)
plt.imshow(image, 'gray')
plt.show()

# modeling
import torch
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=1,
    shuffle=True)

image, label = next(iter(train_loader))
image.shape, label.shape
plt.imshow(image[0, 0, :, :], 'gray')


## 모델링을 위해 필요한 라이브러리
import torch.nn as nn
import torch.nn.functional as F

## convolution
layer = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1).to(torch.device('cpu'))  # GPU 사용할 경우 cuda 로 입력해줘야함
### 참고로 torch.device() 의 값으로 사용할 키워드들은 다음과 같다.
### RuntimeError: Expected one of cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu device type at start of device string: gpu
### 이 외의 값 (ex. GPU, ....)을 넣을 경우 위와같이 RuntimeError 가 발생함

## weight 를 출력할 경우
weight = layer.weight
weight.shape  # weight 는 학습 가능 상태이기 때문에 바로 numpy로 변환이 불가함 -> detach()를 해줘야 출력가능함

weight = weight.detach().numpy()

plt.imshow(weight[0, 0, :, :], 'jet')
plt.colorbar()
plt.show()

output = layer(image).data
res = output.cpu().numpy()
res.shape

image
image_arr = image.numpy()
image_arr.shape

plt.figure(figsize=(15, 30))
plt.subplot(131)
plt.title('Input')
plt.imshow(np.squeeze(image_arr), 'gray')
plt.subplot(132)
plt.title('Weight')
plt.imshow(np.squeeze(weight[0, 0, :, :]), 'jet')
plt.subplot(133)
plt.title('Output')
plt.imshow(np.squeeze(output[0, 0, :, :]), 'gray')
plt.show()

## pooling
image.shape

pool = F.max_pool2d(image, 2, 2)
pool.shape  # 이미지크기가 반감됨

pool_arr = pool.numpy()
pool_arr.shape

plt.figure(figsize=(10, 15))
plt.subplot(121)
plt.title("Input")
plt.imshow(np.squeeze(image_arr), 'gray')
plt.subplot(122)
plt.title("Output")
plt.imshow(np.squeeze(pool_arr), 'gray')
plt.show()

# Linear
image.shape
flatten = image.view(1, 28 * 28)
flatten.shape

linear = nn.Linear(784, 10)(flatten)
linear.shape

plt.imshow(linear.detach().numpy(), 'jet')
plt.show()

# Softmax
with torch.no_grad():
    flatten = image.view(1, 28 * 28)
    linear = nn.Linear(784, 10)(flatten)
    softmax = F.softmax(linear, dim=1)

softmax
np.sum(softmax.numpy())

##
import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np

seed = 1
train_batch_size = 64
test_batch_size = 64
no_cuda = False

use_cuda = not no_cuda and torch.cuda.is_available
device = torch.device('cuda' if use_cuda else 'cpu')

# Preprocess
torch.manual_seed(seed)  # shuffle 시 동일하게 셔플되도록 하기 위한 장치

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=train_batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size,
    shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

## optimization
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.5)

params = list(model.parameters())
for i in range(8):
    print(params[i].size())

model.train()  # 모델을 학습모드로 전환
data, target = next(iter(train_loader))
data.shape, target.shape

data, target = data.to(device), target.to(device)  # device에 탑재

optimizer.zero_grad()  # optimizer 초기화
output = model(data)   # 모델에 데이터 삽입
loss = F.nll_loss(output, target)  # 오차 계산 / loss = negative log-likelihood
loss.backward()  # 역전파 수행
optimizer.step()

## 전체 과정
epochs = 1
log_interval = 100  # 로그 출력을 위한 파라미터

for epoch in range(1, epochs+1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100*batch_idx / len(train_loader), loss.item()))

## evaluation
