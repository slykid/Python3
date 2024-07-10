import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

matplotlib.use('qtagg')
np.random.seed(42)

data = pd.read_csv("Dataset/housing_data/housing.data.txt", sep="\s+", header=None)
data.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

x = data[["LSTAT", "RM", "PTRATIO", "DIS", "NOX", "B", "ZN", "CRIM", "RAD", "TAX"]]
y = data["MEDV"]

# 상수항을 추가합니다.
x = sm.add_constant(x)

# 학습용, 테스트용 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 정규방정식을 적용한 Ridge 회귀
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(x_train, y_train)
y_pred = ridge_reg.predict(x_test)

