import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols

import matplotlib.pyplot as plt
import seaborn as sb

# Q. 어떤 약에 들어간 성분이 평균 10mg 이고, 표준편차가 0.2라고 알려져있다. 성분이 더 적거나 많지는 않은 지를 확인하려한다.
#    성분의 함량을 m 이라고 할 때, 위의 내용을 확인하기 위해, 가설검정에 대한 귀무가설과 대립가설을 작성하시오
#  H0 : m=10  H1 : m!=10

# Q. 어떤 자동차 회사에서 새롭게 개발한 자동차의 연비가 평균 18km 이상이라고 한다.
#    자동차의 평균 연비를 m 이라고 할 때, 위의 가설검정에 대한 귀무가설과 대립가설을 작성하시오.
#  H0 : m >= 18   H1 : m < 18


# Q. 미국 환자 의료 보험비를 선형회귀모델을 이용해 예측하고, 어떤 변수를 사용해야 되는지 최적화된 회귀식을 도출하도록 분석하시오.

# 컬럼 정보
# age : 보험금 수령인의 나이
# sex : 약관자의 성별 / 남성(male) 과 여성(female)로 구성
# bmi : 과체중 혹은 저체중인 사람의 키와 상관관계를 보여주는 신체 용적 지수
# children : 보험에서 보장하는 아이들 수
# smoker : 규칙적인 흡연 여부 / Yes, No로 표시
# region : 미국 내 약관자의 거주지
# charge : 실제 의료보험비

# 데이터 로드
insurance = pd.read_csv("data/insurance.csv")
data = insurance

np.unique(data["region"])

data["sex"] = np.where(data["sex"]=="female", 0, 1)
data["smoker"] = np.where(data["smoker"]=="yes", 1, 0)
data["region"].replace({"northeast":1, "northwest":2, "southeast":3, "southwest":4}, inplace=True)

# 상관관계 시각화
# 연속적 수치를 갖는 변수만 확인
sb.pairplot(data[["age", "bmi", "children", "charges"]])

# 학습, 테스트 데이터 생성
x_train, x_test, y_train, y_test = train_test_split(data[["age", "sex", "bmi", "children", "smoker", "region"]].to_numpy(), data["charges"].to_numpy(),
                                                    test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 모델 성능확인
result = ols(data=data, formula="charges ~ age + sex + bmi + children + smoker + region").fit()
print(result.summary())
# R-squared:                       0.751
# Adj. R-squared:                  0.750
# F-statistic:                     668.1


# 성능개선
x_train, x_test, y_train, y_test = train_test_split(data[["age", "bmi", "children", "smoker", "region"]].to_numpy(), data["charges"].to_numpy(),
                                                    test_size=0.3, random_state=42)

model2 = LinearRegression()
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)

result = ols(data=data, formula="charges ~ .").fit()
print(result.summary())
# R-squared:                       0.751
# Adj. R-squared:                  0.750
# F-statistic:                     802.2

result = ols(data=data, formula="charges ~ age + bmi + children").fit()
print(result.summary())
# R-squared:                       0.120
# Adj. R-squared:                  0.118
# F-statistic:                     60.69


# 최종 모델 선택
# 성능개선
x_train, x_test, y_train, y_test = train_test_split(data[["age", "bmi", "children", "smoker", "region"]].to_numpy(), data["charges"].to_numpy(),
                                                    test_size=0.3, random_state=42)

model2 = LinearRegression()
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
