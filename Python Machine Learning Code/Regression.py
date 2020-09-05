import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

# 연습용 데이터 생성
x = 2 * np.random.rand(100, 1)
y = 3 + x + np.random.randn(100, 1)

# 모델 생성
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_pred = lin_reg.predict(x)

# 회귀선 시각화
plt.plot(x, y_pred, "r-", linewidth=2)
plt.plot(x, y, "b.")
plt.xlabel("x(feature)")
plt.ylabel("y(target)")
plt.show()


# 실습 1. Boston House Sale
# - 타겟
#   1978 보스턴 주택 가격
#   506개 타운의 주택 가격 중앙값 (단위 1,000 달러)

# - 특징
#   CRIM: 범죄율
#   INDUS: 비소매상업지역 면적 비율
#   NOX: 일산화질소 농도
#   RM: 주택당 방 수
#   LSTAT: 인구 중 하위 계층 비율
#   B: 인구 중 흑인 비율
#   PTRATIO: 학생/교사 비율
#   ZN: 25,000 평방피트를 초과 거주지역 비율
#   CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
#   AGE: 1940년 이전에 건축된 주택의 비율
#   RAD: 방사형 고속도로까지의 거리
#   DIS: 직업센터의 거리
#   TAX: 재산세율

# 데이터 로드
origin = load_boston()
print(origin.DESCR)     # 보스턴 주택 매매 데이터 정보 확인
print(origin.__dir__()) # 내부 구성 확인

data = origin.data      # 특징 데이터
target = origin.target  # 타겟 데이터
column_name = origin.feature_names

print(data)
print(target)

df_data = pd.DataFrame(data, columns=column_name)
df_data["Price"] = target
print(df_data.dtypes)   # 데이터프레임 컬럼별 자료형 확인
print(df_data)

# 데이터 탐색
print(df_data.describe())
sns.pairplot(df_data[column_name])
plt.show()

sns.pairplot(df_data[["RM", "LSTAT", "ZN", "Price"]])
plt.show()

# 가설 : 방의 개수는 주택가격에 영향을 줄 것이다.
# 학습데이터, 테스트 데이터 분할하기
# - train : test = 7 : 3 의 비율로 분할

# 방법1. 직접 전체 데이터를 7:3 으로 분할
len(df_data["RM"])  # 506
x_train = df_data["RM"][0 : round(len(df_data["RM"]) * 0.7)]
x_test = df_data["RM"][round(len(df_data["RM"]) * 0.7):len(df_data["RM"])]

y_train = df_data["Price"][0 : round(len(df_data["Price"]) * 0.7)]
y_test = df_data["Price"][round(len(df_data["Price"]) * 0.7):len(df_data["Price"])]

len(x_train)
len(x_test)
len(y_train)
len(y_test)

# 방법2. train_test_split() 으로 분할하기
## 1) array() 함수로 데이터프레임 -> ndarray 형변환 하는 경우
x_train, x_test, y_train, y_test = train_test_split(np.array(df_data["RM"]), np.array(df_data["Price"]),\
                                                    test_size=0.3, random_state=42)

## 2) .to_numpy 를 이용해 데이터프레임 -> ndarray 형변환 하는 경우
x_train, x_test, y_train, y_test = train_test_split(df_data["RM"].to_numpy(), df_data["Price"].to_numpy(),\
                                                    test_size=0.3, random_state=42)

## LinearRegression() 객체는 학습 시 항상 2차원의 데이터를 입력으로 넣어줘야한다.
## 사용한 데이터는 1차원이기 때문에 numpy.newaxis 를 이용해 차원을 1개 증가시켜준다.
x_train = x_train[:, np.newaxis]
x_test = x_test[:, np.newaxis]

# 모델 생성
# - 주택 방 수에 따른 금액의 변화확인
## 모델 객체 생성
model1 = LinearRegression()

# 학습하기
model1.fit(x_train, y_train)

# 예측값 산출하기
y_pred = model1.predict(x_test)

# 회귀선 시각화
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='r')

# 회귀 모형 해석하기
print("회귀계수: ", model1.coef_)
print("상수항: ", model1.intercept_)
print("Fomula : Price ~ " + str(model1.coef_[0]) + " * RM + " + str(model1.intercept_))
# Fomula : Price ~ 9.118102197303786 * RM + -34.662307438406785

# 회귀 모형 성능평가하기
res = sm.ols(data=df_data, formula="Price ~ RM").fit()
res.summary()
# 471.8

# 잔차 분석
# 성능평가
x = df_data.iloc[:, :-1].values
y = df_data['Price'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model = LinearRegression()
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolors='white', label='Training Data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolors='white', label='Test Data')
plt.xlabel("Predicted Values")
plt.ylabel("Residual")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10,50])
plt.tight_layout()
plt.show()

