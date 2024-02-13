import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap

# 결정경계 확인 함수
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

# 데이터 호출
iris = load_iris()
print(iris.DESCR)
print(iris.__dir__())

data = iris["data"]
label = iris["target"]

# 꽃잎 너비로 학습 / iris 종이 Virginica면 1, 아니면 0
x = data[:, 3:]
y = (label==2).astype(np.int)

model = LogisticRegression()
model.fit(x, y)

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = model.predict_proba(x_new)
decision_boundary = x_new[y_prob[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
plt.plot(x[y==0], y[y==0], "bs")
plt.plot(x[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(x_new, y_prob[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(x_new, y_prob[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision\nBoundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Width (cm)", fontsize=14)
plt.ylabel("Prob", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
# plt.save_fig("logistic_regression_plot")
plt.show()

pred = model.predict_prob([[1.7], [1.5]])
print(pred)

# iris data - Logistic Regression
iris = load_iris()

data = pd.DataFrame(iris["data"])
data.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
data["species"] = iris["target"]

x = data.iloc[:, :-1].values
y = (data.iloc[:, -1].values==2).astype(np.int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)

