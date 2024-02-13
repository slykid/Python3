import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

# 1. tensorflow, keras 버전 확인
print(tf.__version__)
print(keras.__version__)


# 2. tensor
## 1) tf.constant : 상수형 텐서
const1 = tf.constant("Hello World")
print(const1)

## tf.constant -> np.ndarray 로 변경하기
arr1 = tf.constant([[1., 2.],
                    [3., 4.]])
print(arr1)
arr1.numpy()

## np.ndarray -> tf.constant 로 변경하기
arr2 = np.array([[1., 2.],
                 [3., 4.]])

print(arr2)

tf_arr2 = tf.convert_to_tensor(arr2)
print(tf_arr2)

### 타입 확인
print(type(arr2))
print(type(tf_arr2))

## 2) 사칙연산
## - Tensor 간의 사칙연산은 element-wise 연산을 기본으로 한다.
## - element-wise(요소별) 연산 : 동일한 위치에 있는 요소간의 연산을 의미한다.
a = tf.ones((2, 2)) * 2
b = tf.ones((2, 2)) * 3
print(a.numpy())
print(b.numpy())

### 기호를 이용한 연산
print(a + b)
print(a - b)
print(a * b)
print(a / b)

### 메소드를 이용한 연산
print(tf.add(a, b).numpy())
print(tf.subtract(a, b).numpy())
print(tf.multiply(a, b).numpy())
print(tf.divide(a, b).numpy())

### numpy 와의 호환이 잘되기 때문에 numpy 값이 들어와도 자동을 변환해줌
print(type(arr2))
tf.multiply(arr2, 10)


## 3) 랜덤값 생성
### tf.random.normal : 표준정규분포를 따르는 상수를 생성한다.
tf.random.normal(shape=(2, 2), mean=0, std=1)
tf.random.normal((2,2), 0, 1)  # 파라미터는 생략가능함
tf.random.uniform(shape=(2,2), minval=0, maxval=1)  # 균등한 분포를 생성할 때는 tf.random.uniform을 사용

## 4) tf.Variable
### 변수생성할 때 사용되는 메소드로, 주로 가중치, 파라미터를 생성할 때 사용된다.
initial_value = tf.random.normal((2, 2), 0, 1)
weight = tf.Variable(initial_value)
print(weight)

### 초기화는 initializer를 사용하는 것도 가능하다.
weight = tf.Variable(tf.random_normal_initializer(stddev=0.1)(shape=(2, 2)))
print(weight)

### variable 값을 갱신할 때는 assign 계열을 사용한다.
new_value = tf.random.normal((2, 2))
print(new_value)
weight.assign(new_value)  # 값을 갱신한다.
print(weight)

new_value = tf.random.normal((2, 2))
print(new_value)
weight.assign_add(new_value)  # 현재 값에 새로들어오는 값을 더한 결과로 갱신한다.
print(weight)

## 5) Broadcasting
### tensor를 사용하는 연산에서 주의할 점은 broadcasting 이다.
### broadcasting 이란 길이 혹은 차원이 서로 다른 백터 간의 사칙연산 시 발생하며,
### 본래는 앞서 언급한 데로, 벡터간에 또는 행렬 간에 element-wise 연산을 통해 사칙연산을 수행하는데 스칼라의 경우
### 자동으로 1-벡터로 변환하여 계산하는 방식을 의미한다.
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

print(matrix1) # tf.Tensor([[3. 3.]], shape=(1, 2), dtype=float32)
print(matrix2) # tf.Tensor(
               # [[2.]
               #  [2.]], shape=(2, 1), dtype=float32)

# 두 개의 배열을 브로드캐스팅하는 것은 다음 규칙을 따릅니다.
# 배열의 랭크가 같지 않으면 두 모양이 같은 길이가 될 때까지 배열의 낮은 랭크쪽에 1을 붙입니다.
# 두 배열은 차원에서 크기가 같거나 배열 중 하나의 차원에 크기가 1 인 경우 차원에서 호환 가능하다고 합니다.
# 배열은 모든 차원에서 호환되면 함께 브로드캐스트 될 수 있습니다.
# 브로드캐스트 후 각 배열은 두 개의 입력 배열의 요소 모양 최대 개수와 동일한 모양을 가진 것처럼 동작합니다.
# 한 배열의 크기가 1이고 다른 배열의 크기가 1보다 큰 차원에서 첫 번째 배열은 마치 해당 차원을 따라 복사 된 것처럼 작동합니다.
print(matrix1+matrix2)
print(matrix1-matrix2)
print(matrix1*matrix2)
print(matrix1/matrix2)

## 6) indexing & slicing
### 2차원 이상인 경우 행, 렬 순서라는 점을 알아 두자
x = tf.constant([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(x)

print(x[0])
print(x[1,1])
print(x[2,3])

## 7) reshape
### 원하는 형태로 벡터를 재구성함
t = tf.constant([[[0, 1, 2],
                [3, 4, 5]],
               [[6, 7, 8],
                [9, 10, 11]]])
print(t.shape)
print(tf.reshape(t, shape=[-1, 3]))  # -1 의 의미는 차원의 크기를 계산해 자동으로 채우라는 의미이다.
                                     # 위의 예제인 경우 3열로 먼저 맞춰주고 나머지부분을 자동으로 채우기 때문에 4행이 되는 것이다.
                                     # 주의 할 점은 형태가 변경된 텐서의 원소 개수는 변경전과 동일하기 때문에 계산을 잘 해서 설정할 것!
print(tf.reshape(t, shape=[-1, 1, 3]))

## 8) expand_dims
### 차원을 하나 증가 시킬 때 사용함
x.shape
tf.expand_dims(x, axis=0).shape  # 행
tf.expand_dims(x, axis=1).shape  # 열
tf.expand_dims(x, axis=-1).shape  # 맨 마지막 차원

## 9) reduce sum/mean
### 특정 차원을 제거한 후 함계/평균 을 계산함
### 축(axis) 옵션을 설정하여 행 또는 열단위로도 연산가능
tf.reduce_sum(x).numpy()
tf.reduce_sum(x, axis=0).numpy()
tf.reduce_sum(x, axis=1).numpy()
tf.reduce_sum(x, axis=-1).numpy()

tf.reduce_mean(x).numpy()
tf.reduce_mean(x, axis=0).numpy()
tf.reduce_mean(x, axis=1).numpy()
tf.reduce_mean(x, axis=-1).numpy()

## 9) argmax
### 계산한 결과 중 가장 큰 값의 "인덱스" 를 반환함
### 기본적으로 축을 포함하여 연산하게 되며, 기본값은 axis = 0 임
tf.argmax(x)
tf.argmax(x, axis=0)
tf.argmax(x, axis=1)
tf.argmax(x, axis=-1)

## 10) One-hot Encoding
### 카테고리 변수의 경우에, 전체 클래스를 대상으로 인덱스에 해당하면 1, 그렇지 않으면 0으로 표기한다.
label = tf.constant([0, 1, 2, 0])
onehot1 = tf.one_hot(label, depth=3) # depth 는 클래스의 개수와 동일함
                                     # 반드시 depth의 값이 label 값의 종류 갯수보다 같거나 커야함
print(onehot1, type(onehot1))


# 3. tf.data
# 주로 데이터 셋을 만들 때 활용된다.
a = np.arange(10)
ds_tensors = tf.data.Dataset.from_tensor_slices(a) # 입력은 ndarray 혹은 list 형식으로 넣어준다.
print(ds_tensors)

## 1) take()
### 데이터 셋 내의 앞 n 개 만큼만 가져오는 경우에 사용
data = ds_tensors.take(5)
for x in data:
    print(x)

## 2) shuffle(), batch()
### shuffle 은 임의로 추출하기 위해서, batch는 한 번에 꺼내올 데이터의 크기를 지정해서 사용함

data = ds_tensors.map(tf.square).shuffle(20).batch(2)   # map : 각 데이터에 계산 함수를 적용하는 경우에 사용한다.
                                                        # tf.square : 값을 제곱해주는 함수
for x in data:
    print(x)

data = ds_tensors.map(tf.square).shuffle(20).batch(3)
for x in data:
    print(x)

## 3) GradientTape
### 주로 그레디언트를 계산할 때 많이 사용되며, tape.watch() 를 통해 모든 연산의 결과를 기록한다.
### 이후 tape.Gradient() 함수를 사용해 미분하여 값을 자동으로 계산해준다.
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)    # x에 나오는 값을 gradient_tape 에 기록됨
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# 입력 텐서 x에 대한 z의 미분
dz_dx = t.gradient(z, x)  # z 를 x 로 미분
print(dz_dx)

### 기본적으로 tf.Variable() 로 선언하게 되면, 자동으로 watch가 적용되기 때문에 별도로 tape.watch() 를 사용할 필요가 없다.
x = tf.Variable(x)
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# 입력 텐서 x에 대한 z의 미분
dz_dx = t.gradient(z, x)
print(dz_dx)


# 3. 선형 회귀
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100, dtype=np.float32)
slope = 1
intercept = np.random.normal(2, 0.2, 100).astype(np.float32)

y = x * slope + intercept

print(slope)
print(intercept)
print(y)

plt.scatter(x, y)
plt.plot(x, x * slope + np.mean(intercept), label="Ground truth", c="r")
plt.legend()
plt.show()

print(x.dtype, y.dtype)
print(x.shape, y.shape)

# dataset으로 생성
## 순서쌍 형식(정확히는 튜플)으로 입력할 경우 대응된 값으로 데이터 셋이 생성됨(같이 셔플되도록 하기 위해서)
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=100).batch(50)

w = tf.Variable(.1, tf.float32)
b = tf.Variable(0., tf.float32)
learning_rate = 0.1

def compute_predictions(x):
    return x * w + b

def compute_loss(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))

def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        predictions = compute_predictions(x)
        loss = compute_loss(y, predictions)
        dloss_dw, dloss_db = tape.gradient(loss, [w, b])

    w.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)
    return loss

loss_list, w_list, b_list = [], [], []
for epoch in range(20):
    sum_loss = 0.
    for x, y in dataset:
        loss = train_on_batch(x, y)
        sum_loss += loss / 2.

    print(epoch+1, "\t", loss.numpy(), "\t", w.numpy(), "\t", b.numpy())
    loss_list.append(loss.numpy())
    w_list.append(w.numpy())
    b_list.append(b.numpy())

plt.plot(loss_list, label="loss")
plt.plot(w_list, label="w")
plt.plot(b_list, label="b")
plt.legend()
plt.show()