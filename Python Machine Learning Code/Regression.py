import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

# 데이터 로드
# 사용 데이터 : 보스턴 데이터
# - 보스턴의 506개 타운(town)의 13개 독립변수값로부터 해당 타운의 주택가격 중앙값을 예측하는 문제
# 컬럼 정보
# CRIM: 범죄율
# INDUS: 비소매상업지역 면적 비율
# NOX: 일산화질소 농도
# RM: 주택당 방 수
# LSTAT: 인구 중 하위 계층 비율
# B: 인구 중 흑인 비율
# PTRATIO: 학생/교사 비율
# ZN: 25,000 평방피트를 초과 거주지역 비율
# CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
# AGE: 1940년 이전에 건축된 주택의 비율
# RAD: 방사형 고속도로까지의 거리
# DIS: 직업센터의 거리
# TAX: 재산세율

data = load_boston()
dir(data)
df_data = pd.DataFrame(data.data, columns=data.feature_names)
df_label = pd.DataFrame(data.target, columns=["MEDV"])

df_data = pd.concat([df_data, df_label], axis=1)
print(df_data.dtypes)
print(df_data)

# 데이터 시각화
fig, ax = plt.subplots(4, 4)
i = 0
for row in range(len(ax)):
    for col in range(len(ax[row])):
        ax[row, col].scatter(df_data[data.feature_names[i]], df_data["MEDV"])
        ax[row, col].set_title(data.feature_names[i])
        i += 1


# 학습 데이터 / 테스트 데이터 분할
# x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(df_data[data.feature_names], df_data["MEDV"], test_size=0.3, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(np.array(df_data[data.feature_names]), np.array(df_data["MEDV"]), test_size=0.3, random_state=42)

model1 = LinearRegression()
model1.fit(pd.DataFrame(x_train["RM"]), y_train)
y_pred = model1.predict(pd.DataFrame(x_test["RM"]))

print("a value : ", model1.intercept_)
print("b balue : ", model1.coef_)

print("MSE (Mean Squared Error : ", mean_squared_error(y_test, y_pred))
print("Coefficient for determination : ", r2_score(y_test, y_pred))
print("Coefficient for determination : ", model1.score(x_test, y_test))

# 결과 시각화
df_data.plot(kind="scatter", x="RM", y="MEDV",
             figsize=(6,6), color="black",
             xlim=(4,8), ylim=(10,45))
plt.plot(x_test["RM"], y_pred, color="blue")

