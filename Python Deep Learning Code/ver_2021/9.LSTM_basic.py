import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

# One-Hot Encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# RNN 계열의 경우, hidden_size, sequence, one-hot vector 를 사용하기 때문에 3차원으로 구성해야한다.
x_data = np.array([[h, e, l, l, o],
                   [e, o, l, l, l],
                   [l, l, e, e, l]], dtype=np.float32)

# 간단하게 LSTM 모델 생성
rnn = layers.LSTM(units=2, return_sequences=False, return_state=True)
outputs, h_states, c_states = rnn(x_data) # 출력 결과 순서 : outputs, hidden_states, cell_states

print('x_data: {}, shape: {} \n'.format(x_data, x_data.shape))
print('outputs: {}, shape: {} \n'.format(outputs, outputs.shape))
print('hidden_states: {}, shape: {}'.format(h_states, h_states.shape))
print('cell_states: {}, shape: {}'.format(c_states, c_states.shape))