import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
X3D_inv = pca.inverse_transform(X2D)

axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)

C = pca.components_
R = C.T.dot(C)
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 3.8))
ax = fig.add_subplot(111, projection='3d')

X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]

ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
np.linalg.norm(C, axis=0)
ax.add_artist(Arrow3D([0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.add_artist(Arrow3D([0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.plot([0], [0], [0], "k.")

for i in range(m):
    if X[i, 2] > X3D_inv[i, 2]:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
    else:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#505050")

ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
ax.set_xlabel("$x_1$", fontsize=18, labelpad=7)
ax.set_ylabel("$x_2$", fontsize=18, labelpad=7)
ax.set_zlabel("$x_3$", fontsize=18, labelpad=4)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

ax.plot(X2D[:, 0], X2D[:, 1], "k+")
ax.plot(X2D[:, 0], X2D[:, 1], "k.")
ax.plot([0], [0], "ko")
ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.set_xlabel("$z_1$", fontsize=18)
ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
ax.axis([-1.5, 1.3, -1.2, 1.2])
ax.grid(True)

# Swiss roll 예제 시각화
from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18, labelpad=7)
ax.set_ylabel("$x_2$", fontsize=18, labelpad=7)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis(axes[:4])
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0, labelpad=10)
plt.grid(True)

plt.subplot(122)
plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)
plt.show()

# PCA 모델링
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

W2 = Vt.T[:, 2]
X2D = X_centered.dot(W2)
print(X2D)

# 원본(X_centered)와 비교
print(X_centered.shape)  # (60, 3)
print(X_centered[:, 0])
print(X_centered[:, 1])
print(X_centered[:, 2])

pca = PCA(n_components=2)  # 차원을 2차원으로 줄이는 모델
x2d = pca.fit_transform(X)  # 모델 학습
print(pca.components_)
print(pca.components_.T)

print(pca.explained_variance_ratio_)

# 적정 차원 수 계산
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1)
X = mnist["data"]
Y = mnist["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95)
print(d, cumsum[d])

plt.plot(range(1,785), cumsum)
plt.xlim(0, 400)
plt.axhline(cumsum[d], color='r', linewidth=1, linestyle='--')
plt.axvline(d, color='r', linewidth=1, linestyle='--')
plt.plot(d, cumsum[d], color='r' , marker='o')
plt.annotate("("+str(d)+", "+ str(cumsum[d])+")" , xy=(d,cumsum[d]) , xytext=(d+5, cumsum[d]-5), size=15)
plt.show()

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
print(pca.n_components_)
np.sum(pca.explained_variance_ratio_)

pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("원본", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("압축 후 복원", fontsize=16)
plt.savefig("images/dimension_reduction/mnist_compression_plot")

# 커널 PCA
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

# 스위스롤 적용
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

linear_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sigmoid_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

y = t > 6.9

plt.figure(figsize=(11, 4))
for subplot, pca, title in (
(131, linear_pca, "선형 커널"), (132, rbf_pca, "RBF 커널, $\gamma=0.04$"), (133, sigmoid_pca, "시그모이드 커널, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    if subplot == 132:
        X_reduced_rbf = X_reduced

    plt.subplot(subplot)
    # plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
    # plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
plt.savefig("images/dimension_reduction/kernel_pca_plot")
plt.show()

# 커널 선택 및 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
y = t > 6.9

clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression(solver='liblinear'))
])
param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
print(grid_search.best_params_)  # {'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

from sklearn.metrics import mean_squared_error
print("MSE between original to prediction : ", mean_squared_error(X, X_preimage))

# LLE
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)

# 실습 1 Wine Data 주성분 추출
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터셋 로드
wine = pd.read_csv("https://archive.ics.uci.edu/ml/" 
                   "machine-learning-databases/wine/wine.data", header=None)

# 훈련, 테스트용 데이터 셋 각각 생성
# 컬럼 수 : 13개
x, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values

# 7:3 으로 데이터 split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)

# 표준화 처리
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

# 공분산 계산
cov = np.cov(x_train_std.T)  # 공분산 행렬 계산
eigen_vals, eigen_vector = np.linalg.eig(cov)  # 고유값 분해 -> 고유 벡터, 고유값

print('고유값 \n%s' % eigen_vals)

total = sum(eigen_vals)
exp_val = [(i / total) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(exp_val)

plt.bar(range(1, 14), exp_val, alpha=0.5, align='center', label='Individual Explained Varience')
plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative Explained Varience')
plt.ylabel('Explained Varience Ratio')
plt.xlabel('Principal Component Index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
plt.savefig('images/dimension_reduction/Variance Sum to Component Index.png')

# 고유 벡터 - 고유값 쌍(튜플형식)으로 구성 후, 내림차순으로 정렬
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vector[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
print(eigen_pairs)

# 투영 행렬 W 생성
# 정렬된 것 중에서 가장 큰 고유값 2개와 쌍을 이루는 고유 벡터들을 선택한다. (전체 분산의 약 60% 정도를 사용할 것으로 예상됨)
# 투영 행렬은 13 x 2  형식의 리스트로 저장함
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))

print(w)

# PCA 부분공간 산출하기
# X' = XW ( 부분공간 : X' / X : 원본 , W : 투영행렬 )

# x_train_std[0].dot(w)  # array([2.38299011, 0.45458499])
x_train_pca = x_train_std.dot(w)

# 변환된 훈련데이터의 시각화
colors = ['r', 'g', 'b']
markers = ['s', 'x', 'o']
for label, color, marker in zip(np.unique(y_train), colors, markers):
    plt.scatter(x_train_pca[y_train==label, 0],
                x_train_pca[y_train==label, 1],
                c=color, label=label, marker=marker)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
plt.savefig('images/dimension_reduction/wine_dim_reduction_result.png')


# LDA (Linear Discriminant Analysis)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터셋 로드
wine = pd.read_csv("https://archive.ics.uci.edu/ml/" 
                   "machine-learning-databases/wine/wine.data", header=None)

# 훈련, 테스트용 데이터 셋 각각 생성
# 컬럼 수 : 13개
x, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values

# 7:3 으로 데이터 split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)

# 표준화 처리
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

# 평균 벡터를 이용한 산포행렬 계산
# 실행전 y 의 label 갯수 확인
y_label = set(y_train)
print(y_label) # 1 ~ 3

np.set_printoptions(precision=4)  # 부동소수점, Array, 기타 numpy 객체가 표시되는 방식을 설정함
mean_vecs =[]
for label in range(1, 4):
    mean_vecs.append(np.mean(x_train_std[y_train == label], axis=0))  # 각 label 별 평균값을 계산
    print("Mean Vector %s : %s\n" % (label, mean_vecs[label-1]))

d = 13  # 특성 계수
Sw = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in x_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    Sw += class_scatter
print("클래스 내의 산포 행렬 : %s x %s" % (Sw.shape[0], Sw.shape[1]))
print('클래스 레이블 분포 : %s' % np.bincount(y_train)[1:])

Sw = np.zeros(((d, d)))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(x_train_std[y_train == label].T, bias=True)
    Sw += class_scatter
print("스케일 조정된 클래스 내의 산포 행렬 : %s x %s" % (Sw.shape[0], Sw.shape[1]))

mean_class = np.mean(x_train_std, axis=0)
mean_class = mean_class.reshape(d, 1)
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = x_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    S_B += n * (mean_vec - mean_class).dot((mean_vec - mean_class).T)
print("클래스 간 산포행렬 : %s x %s" % (S_B.shape[0], S_B.shape[1]))

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print("내림차순 고유값\n")
for eigen_val in eigen_pairs:
    print(eigen_val[0])

total = sum(eigen_vals.real)
discrimin = [(i / total) for i in sorted(eigen_vals.real, reverse=True)]
cum_discrimin = np.cumsum(discrimin)
plt.bar(range(1, 14), discrimin, alpha=0.5, align="center", label="Individual Discriminability")
plt.step(range(1, 14), cum_discrimin, where="mid", label="Cumulative Discriminability")
plt.ylabel("Discriminability ratio")
plt.xlabel("Linear Discriminants")
plt.ylim([-0.1, 1.1])
plt.legend(loc="best")
plt.tight_layout()
plt.show()

W = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print("행렬 W:\n", W)

x_train_lda = x_train_std.dot(W)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(x_train_lda[y_train==l, 0],
                x_train_lda[y_train==l, 1] * (-1),
                c=c, label=l, marker=m)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap

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
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

x_test_lda = lda.transform(x_test_std)
plot_decision_regions(x_test_lda, y_test, classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()