import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
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
