import os
import time
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 시각화 시 한글 깨짐에 대한 처리
# matplotlib.matplotlib_fname()
# font_list = sorted([font.name for font in fm.fontManager.ttflist])
# font_list
#
# print ('버전: ', matplotlib.__version__)
# print ('설치위치: ', matplotlib.__file__)
# print ('설정: ', matplotlib.get_configdir())
# print ('캐시: ', matplotlib.get_cachedir())

matplotlib.rc('font', family='NanumBarunGothic')

# 이미지 저장을 위한 경로 설정 및 폴더 생성
PROJECT_ROOT_DIR = "D:\\workspace\\Python3"
CHAPTER_ID = "svm"
if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is True:
    if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)) is False:
        os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))
    else :
        time.sleep(1)
elif os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is False:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images"))
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))
else:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))

# 시각화 저장 함수
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # 결정 경계에서 w0*x0 + w1*x1 + b = 0 이므로
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

# 실습 1. Iris-Virginica 분류하기
# iris 데이터 로드
iris = load_iris()
print(iris.feature_names)

df_iris = pd.DataFrame(iris["data"], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
df_iris["species"] = iris["target"]
print(df_iris)

df_iris.describe()

# 표준화
# - SVM 은 이상치와 특성 스케일에 민감하다
scaler = StandardScaler()
df_iris_scaler = pd.DataFrame(scaler.fit_transform(df_iris.iloc[:, 0:4]), columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

# Iris-Virginica 여부를 확인하기 위한 Flag 생성
# - Iris-Virginica = 2 임
df_iris_scaler["virginica_YN"] = (df_iris["species"]==2).astype(np.float64) # Y = 1 / N = 0

# 학습, 테스트 용 데이터 생성
# 학습 : 테스트 = 7 : 3
x_train, x_test, y_train, y_test = train_test_split(df_iris_scaler.iloc[:, 0:4].to_numpy(),
                                                    df_iris_scaler["virginica_YN"].to_numpy(),
                                                    test_size=0.3, random_state=42)

# 모델 학습
# 일반적으로 SVM 은 선형 SVM이므로 'linear' 모델로 생성
# 파라미터 C : 비용(Cost) 를 의미, 얼마나 많은 데이터 샘플이 다른 클래스에 놓이는 것을 허용하는지를 결정함
model = SVC(kernel='linear', C=10**9)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 결과 확인
print(y_pred==y_test)
# [ True  True  True  True  True  True  True False  True  True  True  True
#   True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True  True]

count = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        count += 1
print("일치 개수 : ",  str(count) + "/" + str(len(y_test)))
# 일치 개수 :  44/45


# 커널 SVM (가우시안 커널)
import os
import time
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons
from sklearn.svm import SVC

import matplotlib
import matplotlib.pyplot as plt

# 시각화 글꼴 설정
matplotlib.rc('font', family='NanumBarunGothic')

# 이미지 저장 폴더 생성
PROJECT_ROOT_DIR = "D:\\workspace\\Python3"
CHAPTER_ID = "svm"
if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is True:
    if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)) is False:
        os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))
    else :
        time.sleep(1)
elif os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is False:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images"))
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))
else:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))

# 시각화 저장 함수
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# 시각화 함수
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

# 데이터 로드
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# 하이퍼파라미터 설정
gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

# 모델 설정
# - 모델 생성 및 학습의 편의를 위해 파이프라인 구축
svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))  # 가우시안 RBF 커널을 사용할 것이므로 'rbf' 를 커널에 설정
        ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

# 학습 결과 시각화
plt.figure(figsize=(11, 7))
for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(221 + i)
    plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
save_fig("moons_rbf_svc_plot")
plt.show()