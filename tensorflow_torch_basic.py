# 1. tensorflow 사용법
import numpy as np
import pandas as pd
import tensorflow as tf

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

plt.title(train_y[0])
plt.imshow(train_x[0], "gray")
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


# 2. torch 사용법
import torch

# torch 버전 확인
torch.__version__