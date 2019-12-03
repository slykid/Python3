# 1. tensorflow 사용법
import numpy as np
import pandas as pd
import tensorflow as tf

## 1) Tensor 생성
# numpy 에서의 array
a = np.array([1, 2, 3])
a.shape

# tensorflow의 tensor
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

# 난수 생성
np.random.randn(9)  # numpy 에서 정규분포를 갖는 난수 생성방법
tf.random.normal([3, 3])  # 정규 분포를 갖는 난수 생성함(normal distribution) / numpy 와 달리 shape를 넣어 줘야 함
tf.random.uniform([3, 3])  # 정규 분포와 관계없이 난수를 생성함(uniform distribution) / numpy 와 달리 shape를 넣어 줘야 함

# 데이터 불러오기
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import datasets  # 텐서플로우에서 제공하고 있는 데이터셋 라이브러리

## mnist 데이터 로드
mnist = datasets.mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x.shape  # 데이터 확인

image = train_x[0]
image.shape  # Out[9]: (28, 28) 입력 뒤에 3이 없다는 의미는 grayscale 의 이미지라는 뜻
plt.imshow(image, 'gray')  # 이미지 출력





# 2. torch 사용법
import torch

# torch 버전 확인
torch.__version__