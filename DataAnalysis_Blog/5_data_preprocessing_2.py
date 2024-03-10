import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()

# 1. 정규화
x = iris.data
y = iris.target

print("변경 전: ")
print(x)

scaler = MinMaxScaler()
x_norm = scaler.fit_transform(x)

print("변경 후")
print(x_norm)

# 2. 표준화 vs. 정규화
sample = np.array([0,1,2,3,4,5])

print("표준화 결과: ", (sample - sample.mean()) / sample.std())
print("정규화 결과: ", (sample - sample.min()) / sample.max() - sample.min())

# 3. 정규화
scaler = StandardScaler()
x_standard = scaler.fit_transform(x)
print(x_standard)

