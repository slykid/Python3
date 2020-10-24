# 작성일자: 2020.10.17
# 작성자: 김 길 현

# 1. PCA
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, KernelPCA

## 데이터 생성
### 가중치 및 편향 설정
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

### 원본데이터 생성
### - 3차원의 데이터를 생성
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

pca = PCA(n_components=2)
x2d = pca.fit_transform(X)
print(x2d)  # 투영된 결과
print(pca.components_.T)  # 주성분의 벡터 값 / 출력 결과가 행 벡터이기 때문에 전치 작업 필수
print(pca.explained_variance_)  # 각 주성분을 설명하기 위한 변수별 분산 값
print(pca.explained_variance_ratio_)  # 각 주성분을 설명하기 위한 변수별 분산 비중

kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
x2d_reduce = kernel_pca.fit_transform(X)
print(x2d_reduce)

# 2. LDA
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

