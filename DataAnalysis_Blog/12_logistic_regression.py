import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

matplotlib.use("qtagg")

iris = load_iris()
features = iris.data
targets = iris.target
targets = targets.reshape(-1, 1)

plt.figure(figsize=(18,8),dpi=100)
plt.scatter(features.T[0],features.T[2])
plt.title('IRIS Petal and sepal length', fontsize=20)
plt.ylabel('Petal Length')
plt.xlabel('sepal length')
plt.savefig("/Users/kilhyunkim/Pictures/iris_data.jpg")

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)

# 1. 단순선형회귀로 분류문제 도전하기
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

# 잔차 계산
residuals = model.resid

# 잔차 그래프 그리기
plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted values')
plt.show()
plt.savefig("/Users/kilhyunkim/Pictures/iris_linear_regression.jpg")

# 잔차의 정규성 검토 - Q-Q plot
sm.qqplot(residuals, line='45')
plt.title('Q-Q plot of residuals')
plt.show()
plt.savefig("/Users/kilhyunkim/Pictures/iris_linear_regression-qq.jpg")

# 잔차의 히스토그램
plt.hist(residuals, bins=30, edgecolor='k')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of residuals')
plt.show()
plt.savefig("/Users/kilhyunkim/Pictures/iris_linear_regression-hist.jpg")

# 잔차의 통계 요약
print(f'Residuals mean: {np.mean(residuals):.4f}')
print(f'Residuals variance: {np.var(residuals):.4f}')

# 2. 로지스틱 회귀
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

matplotlib.use("qtagg")

iris = load_iris()

x = iris['data'][:,3:]
y = (iris['target']==2).astype(np.int32)

model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(x, y)

x_n = np.linspace(0,3,100).reshape(-1,1)
y_pred = model.predict_proba(x_n)
y_pred

decision_boundary = x_n[y_pred[:,1]>=0.5][0]
print("Decision Boundary: " + str(decision_boundary[0]))

plt.figure(figsize=(8,3))
plt.plot(x[y==0],y[y==0],'bs')
plt.plot(x[y==1],y[y==1],'g^')
plt.plot([decision_boundary, decision_boundary],[-1,2],'k:',linewidth=2)
plt.plot(x_n, y_pred[:, 1], 'g-', linewidth=2, label='Iris-Virginica')
plt.plot(x_n, y_pred[:, 0], 'b--', linewidth=2, label='Not Iris-Virginica')
plt.text(decision_boundary[0]+0.02, 0.15, 'Decision Boundary', fontsize=14, color='k', ha='center')
plt.arrow(decision_boundary[0], 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary[0], 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel('petal width(cm)', fontsize=14)
plt.xlabel('probability        ', fontsize=14, rotation=0)
plt.legend(loc='center left', fontsize=14)
plt.axis([0,3,-0.02,1.02])
plt.show()
plt.savefig("/Users/kilhyunkim/Pictures/iris_is-Virginica.jpg")

# 판별모델 생성
X = iris['data'][: , (2, 3)]
y = (iris['target'] == 2).astype(np.int32)

log_reg = LogisticRegression(solver='liblinear', C=10**10, random_state=42)
log_reg.fit(X,y)

x0, x1 = np.meshgrid(
                np.linspace(2.9, 7, 500).reshape(-1,1),
                np.linspace(0.8, 2.7, 200).reshape(-1,1),
                )
X_n = np.c_[x0.ravel(), x1.ravel()]
y_p = log_reg.predict_proba(X_n)

plt.figure(figsize=(10,4))
plt.plot(X[y==0, 0], X[y==0, 1], 'bs')
plt.plot(X[y==1, 0], X[y==1, 1], 'g^')

zz = y_p[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0] / log_reg.coef_[0][1])

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, 'k--', linewidth=3)
plt.text(3.5, 1.5, 'Not Iris_Virginica', fontsize=14, color='b', ha='center')
plt.text(6.5, 2.3, 'Iris_Virginica', fontsize=14, color='g', ha='center')
plt.xlabel('Petal Length', fontsize=14)
plt.ylabel('Petal Width                       ', fontsize=14, rotation=0)
plt.axis([2.9, 7, 0.8, 2.7])
plt.show()
plt.savefig("/Users/kilhyunkim/Pictures/iris_decision_linear.jpg")

