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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap

wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
x, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)

# 표준화 처리
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)


lda = LDA(n_components=2)
x_train_lda = lda.fit_transform(x_train_std, y_train)

def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z, alpha=.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = x[y == cl, 0], y = x[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx],
                    label=cl, edgecolors="black")

    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]

        plt.scatter(x_test[:, 0], x_test[:, 1],
                    c='', edgecolors="black", alpha=1.0,
                    linewidth=1, marker="o",
                    s=100, label="test set")

lr = LogisticRegression(solver="liblinear", multi_class="auto")
lr = lr.fit(x_train_lda, y_train)
plot_decision_regions(x_train_lda, y_train, classifier=lr)

x_test_lda = lda.transform(x_test_std)
plot_decision_regions(x_test_lda, y_test, classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

