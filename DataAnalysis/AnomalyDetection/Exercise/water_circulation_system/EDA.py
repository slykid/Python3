# [시나리오]
# 1. 문제정의
# - 사업 확장으로 인한 관리포인트 증가에 따른 운영 및 유지보수 자원 부족
# - Water Circulation System 고장으로 인한 리스크 발생

# 2. 기대효과
# - 이상진단 시스템을 통한 사전 유지보수를 통해 고장 발생으로 인한 리스크 감소

# 3. 해결방안 - 모델링을 통한 이상진단시스템 구축 및 운영
# 1) 데이터 전처리 & EDA
# 2) 시계열 센서 데이터 분석
# 3) 이상탐지 모델링

# 4. 성과측정
# - 이상진단 시스템 활용 전/후 고장 발생률 비교
# - 이상진단 솔루션 운영 전/후 신규 고객 증가 및 기존 고객 만족도 조사

import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# 전역변수 및 설정
matplotlib.use("MacOSX")
# matplotlib.use("QtAgg")
plt.style.use(["dark_background"])

# 데이터 로드
df = pd.read_csv("Dataset/anomaly_detection/water_circulation_system/chapter01_df.csv", sep=";")
df.head()

# 1. 데이터 전처리 및 EDA
# 1.1 데이터 전처리
print("Shape:", df.shape)
print(df.info())

print("Null Value Count:", df.isnull().sum())
print(df.describe())

sns.distplot(df['Pressure'])

# 1.2 EDA
# 1.2.1 Anormaly 및 Change Point 확인
pd.DataFrame({"count": df["anomaly"].value_counts(), \
              "ratio(%)": df["anomaly"].value_counts(normalize=True) * 100})

pd.DataFrame({"count": df["changepoint"].value_counts(), \
              "ratio(%)": df["changepoint"].value_counts(normalize=True)* 100})

# 1.2.2 Trend Analysis
df_anormaly = df[df["anomaly"] == 1]
df_changepoint = df[df["changepoint"] == 1]
df_normal = df[df["changepoint"] == 0]

plt.figure(figsize=(24, 5))
plt.plot(df_anormaly.index, df_anormaly["Accelerometer1RMS"], 'o', color='red', markersize=5)
plt.plot(df_changepoint.index, df_changepoint["Accelerometer1RMS"], 'o', color='green', markersize=5)
plt.plot(df_normal.index, df_normal["Accelerometer1RMS"], linestyle='--', color='grey')

# 2. 시계열 센서 데이터 분석
# 2.1 센서 별 시각화
# - 다른 변수들간 이상치에 대한 상관관계 여부를 파악
df.columns[1:9], len(df.columns[1:9])

for v, i in enumerate(df.columns[1:9]):
    plt.figure(figsize=(24, 15))
    # plt.subplot(7, 1, v+1)
    plt.plot(df_anormaly.index, df_anormaly[i], 'o', color='red', markersize=5)
    plt.plot(df_changepoint.index, df_changepoint[i], 'o', color='green', markersize=5)
    plt.plot(df_normal.index, df_normal[i], linestyle='--', color='grey')
    plt.title(i)
plt.show()
# 단일 변수만으로는 특별히 이상 현상이 없음 -> 변수들을 복합적으로 활용해 확인 필요

# 2.2 센서별 분포
n_cols = 3
n_rows = 3

fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(20, n_rows*5))

for i, col in enumerate(df.columns[1:9]):
    sns.distplot(df_normal[col], ax=ax[int(i / n_cols), int(i % n_cols)])
    sns.distplot(df_anormaly[col], ax=ax[int(i / n_cols), int(i % n_cols)])

# 2.3 변수간 상관관계 분석
df_corr = df.iloc[:, 1:-1]

fig = plt.figure(figsize=(8, 8))
df_num = df_corr.corr()
sns.heatmap(df_num, vmin=-1, vmax=+1, annot=True, cmap="coolwarm", linewidths=0.5, mask=np.triu(df_num.corr()))

# 3. 이상탐지 모델링
# 3.1 Model Selection
# 3.1.1 Isolation Forest: 다변량에서 효과적

X = df.drop(["datetime", "anomaly", "changepoint"], axis=1)
y = df["anomaly"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1234)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train shape: {y_test.shape}")

pd.Series(y_train).value_counts(normalize=True)
pd.Series(y_test).value_counts(normalize=True)

clf = IsolationForest(max_samples=200, contamination=0.3, random_state=1234)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

y_pred_train = np.where(y_pred_train == -1, 1, 0)
y_pred_test = np.where(y_pred_test == -1, 1, 0)

# 모델 성능평가
print(classification_report(y_train, y_pred_train))
print("------------------------------------------------------------------")
print(classification_report(y_test, y_pred_test))

# 3.2 Scoring 기반 Threshold 조정
y_pred_train[0:5], clf.decision_function(X_train)[0:5]

sns.distplot(clf.decision_function(X_train), label="Train")
sns.distplot(clf.decision_function(X_test), label="Test")
plt.legend()

# 3.3 Threshold 재조정
# 3.3.1 score 변수 할당
## 수치 비교를 하면서 조절해주면 됨
y_pred_train_score = clf.decision_function(X_train)
y_pred_test_score = clf.decision_function(X_test)

y_pred_train = np.where(y_pred_train_score < 0.05, 1, 0)
y_pred_test = np.where(y_pred_test_score < 0.05, 1, 0)

print(classification_report(y_train, y_pred_train))
print("------------------------------------------------------------------")
print(classification_report(y_test, y_pred_test))
