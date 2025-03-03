# [문제 상황]
# A사는 반도체를 생산하는 글로벌 회사인데, 반도체 구성품 중 Wafer는 반도체 집적회로의 핵심 재료이다.
# A사는 반도체의 성능을 향상시키기 위해서 최근 Wafer 설계를 변경하고 제품을 생산 중에 있으며, 설계 변경 과정에서 불량 제품이 발생하고 있다.
# 이에 대해 이상탐지 모델링을 통해 이상인 Wafer를 사전에 검출하고자 한다.

import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import classification_report

matplotlib.use("MacOSX")
plt.style.use(['dark_background'])
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

data_path = "/Users/kilhyunkim/workspace/Python3/DataAnalysis/Dataset/anomaly_detection/semi-conductor_production"

df_train = pd.read_csv(data_path + "/train.csv")
df_test = pd.read_csv(data_path + "/test.csv")

data = pd.concat([df_train, df_test], axis=0)
data.head()

# 데이터 명세
# - feature 1 ~ n: Wafer의 특성 데이터
# - Class: 이상 유/무 (0: 정상, 1: 이상)
print(data.info())
print(data.info)
data["Class"].value_counts()

# 1. 문제해결 프로세스 정의
# 1.1 문제정의
# - 최근 설계 변경으로 인한 Wafer 불량 발생

# 1.2 기대효과
# - Wafer 불량 사전 탐지를 통해 반도체 완성 전 사전 처리
# - 불량 및 폐기 비용 감소

# 1.3 해결 방안
# - 이상탐지 모델링을 통한 반도체 완성품 조립 전 Wafer 불량 탐지
#   - 데이터 전처리 및 EDA
#   - Feature Selection
#   - 이상탐지 모델링

# 1.4 성과 측정
# - 모델 활용 전/후 Wafer 불량률 비교

# 1.5 현업 적용
# - Wafer 공정 데이터 수집 체계 구축
# - 공정 데이터 Model Input
# - 이상 Wafer 추출 및 점검

# 2. 데이터 전처리
# 2-1. 데이터 확인
print("data: ", data.shape)
print(data.info)

# 2-2. NULL 값 확인
pd.set_option("display.max_rows", 2000)
print(data.isnull().sum())

## Class 컬럼의 NaN 값 확인
print(data[data['Class'].isna()].head())

## NaN 데이터 삭제
data = data.dropna()
print(data.isnull().sum().sum())  # 단순 sum을 하면 컬럼별 null 값의 갯수가 나옴

# 2-3. Outlier 확인
print(pd.Series(pd.DataFrame(data.describe()).loc["min"] < 0).value_counts())


# 3. EDA
# 3-1. 데이터 특성 파악을 위한 초기 탐색
print(data["Class"].value_counts())

# 3-2. Normal vs. Abnormal 측정값 평균 비교, 현재 가진 데이터로 분류가능한지 가늠
data_normal = data[data["Class"] == 0]
data_abnormal = data[data["Class"] == 1]

data_normal_mean = pd.DataFrame(data_normal.describe()).loc["mean"]
data_abnormal_mean = pd.DataFrame(data_abnormal.describe()).loc["mean"]

## 더 정확하게 비교하기 위해 표준화 작업 수행
data_concat = pd.concat([data_normal_mean, data_abnormal_mean], axis=1, keys=["normal", "abnormal"])
data_concat["gap"] = abs(data_concat["normal"] - data_concat["abnormal"])
data_concat.head()

data_top10 = data_concat.sort_values(by=["gap"],ascending=False).head(10)
print(data_top10)  # feature2 의 이상 수치가 압도적으로 높음


# 3-3. Feature Selection
X = data.drop(["Class"], axis=1)
y = data["Class"]

# 3-3-1. 정규화
X_ = StandardScaler().fit_transform(X)

data_s = pd.concat([pd.DataFrame(X_, columns=X.columns)
                   , pd.DataFrame(y, columns=["Class"])]
                   , axis=1)
print(data_s.head())

# 3-3-2. Normal vs. Abnormal 측정값 평균 비교
data_s_normal = data_s[data_s["Class"] == 0]
data_s_abnormal = data_s[data_s["Class"] == 1]

data_s_normal_mean=pd.DataFrame(data_s_normal.describe()).loc["mean"]
data_s_abnormal_mean=pd.DataFrame(data_s_abnormal.describe()).loc["mean"]

print("Normal mean: ", data_s_normal_mean)
print("Abnormal mean: ", data_s_abnormal_mean)

data_concat = pd.concat([data_s_normal_mean, data_s_abnormal_mean], axis=1, keys=["normal", "abnormal"])
data_concat["gap"] = abs(data_concat["normal"] - data_concat["abnormal"])
data_concat.head()

data_top10 = data_concat.sort_values(by=["gap"],ascending=False).head(10)
print(data_top10)  # 기존 feature2 와 달리, 표준화를 한 결과 feature_1400 의 이상 수치가 더 높게 나왔다.


# 3-4. 연속형 변수 구간화
data_s_des = pd.DataFrame(data_s.describe())
data_s_des

# 3-4-1. bin 구간 생성 및 np.digitize -> level 부여
for i, col in enumerate(data_s_des.columns[:-1]):
    bins = [data_s_des.loc["min"][i], data_s_des.loc["25%"][i], data_s_des.loc["75%"][i], np.inf]
    feature_nm = col + "_gp"
    data_s[feature_nm] = np.digitize(data_s[col], bins)

data_s["Class"].value_counts(normalize=True)
data_s.columns[0:1559]
data_s.columns[1559:]

# 3-4-2. 향상도(Lift) 계산
result_list = []
target_ratio = 8.11

for i in data_s.columns[0:1558]:
    gp = i + "_gp"
    tmp = data_s.groupby(gp)["Class"].agg(["count", "sum"])

    # groupby 기준의 불량률, 향상도 계산
    tmp["ratio"] = round((tmp["sum"] / tmp["count"]) * 100, 2)
    tmp["lift"] = round(tmp["ratio"] / target_ratio, 2)  # 1 이상의 수치: 이상일 확률이 높음 / 1 이하의 수치: 정상일 확률이 높음

    df_tmp = pd.DataFrame(tmp)

    gap = df_tmp["lift"].max() - df_tmp["lift"].min()
    df_loop = pd.DataFrame([[i, gap]], columns=["val", "gap"])
    result_list.append(df_loop)

    df_accum_start = pd.concat(result_list)

print(len(df_accum_start))
df_accum_start.sort_values(by=["gap"], ascending=False).head(10)

# 확인하기
# gp = "feature_2_gp"
gp = "feature_1026_gp"  # 고정값이여서 하나의 그룹만 잡힘

tmp = data_s.groupby(gp)["Class"].agg(["count", "sum"])
tmp["ratio"] = round((tmp["sum"] / tmp["count"]) * 100, 2)
tmp["lift"] = round(tmp["ratio"] / target_ratio, 2)
print(tmp)


# 4. 이상 탐지 모델링
# 4-1. Model Selection
# - 주요 변수의 개수가 많이 없음 > PCA + Mahalanobis Distance

# [장점]
# - 데이터 분포를 고려한 이상 탐지
# - 비선형 관계의 데이터에 사용 가능
# - 데이터 자체에 대한 가정이 필요없음

# [단점]
# - 변수간의 관계가 독립 = 유클리드 거리와 동일
# - 변수간의 상관성이 명확하지 않은 경우 적용이 어려움 ("마할라노비스 거리 = 공분산을 고려한 알고리즘" 이기 때문)

raw  = data_s.iloc[:, 0:1559]
raw.head()

X = raw.drop(["Class"], axis=1)
y = raw["Class"]

pca = PCA()
pca.fit(X)
reduced = pca.transform(X)

features = range(pca.n_components_)
df_features = pd.DataFrame(features, columns=["pc_feature"])

df_variance = pd.DataFrame(data=pca.explained_variance_ratio_, columns=["variance"])

df_pc_variance = pd.concat([df_features, df_variance], axis=1)
df_pc_variance.head()  # PC0, PC1으로도 전체 분산의 3%(0.03xxx) 정도 밖에 표현하지 못함


# feature 별 분산을 확인하기 위한 시각화
fig, ax = plt.subplots(figsize=(10, 4))
xi = np.arange(1, reduced.shape[1]+1, step=1)
yi = np.cumsum(pca.explained_variance_ratio_) * 100

plt.ylim(0.0, 101)
plt.xlabel("Number of Principle Components")
plt.ylabel("Explained Features (%)")
sns.scatterplot(x=xi, y=yi)
plt.show()

# 차원 축소 데이터 셋 생성
pca = PCA(n_components=400)
pc = pca.fit_transform(X)

df_pc = pd.DataFrame(data=pc).reset_index(drop=True)
df_pc.head()

# 마할라노비스 거리함수 모델 생성
detector = EllipticEnvelope(contamination=.10)
detector.fit(df_pc)

pred = detector.predict(df_pc)
pd.Series(pred).value_counts()  # -1 = Outlier

# 평가를 위한 재 라벨링 수행
y_pred = np.where(pred == -1, 1, 0) # 1 = Outlier / 0 = Normal

# 성능평가
print(classification_report(y, y_pred))
#       precision    recall  f1-score   support
# 0.0       0.93      0.91      0.92      1620
# 1.0       0.22      0.27      0.24       143
# accuracy                           0.86      1763
# macro avg       0.58      0.59      0.58      1763
# weighted avg       0.88      0.86      0.87      1763


answer = pd.DataFrame({ "pred" : y_pred, "Class": y }, columns=["pred", "Class"])
answer[answer["pred"]==1].head()


# Scoring 기반 Threshold 조정
detector.dist_

fig, ax = plt.subplots(figsize=(25, 5))
sns.displot(detector.dist_, label="distrubution")
plt.legend()
plt.show()

0.1 * pow(10, 8)

y_pred_re = np.where(detector.dist_ > 0.1 * pow(10, 8), 1, 0)
print(classification_report(y, y_pred_re))
#        precision    recall  f1-score   support
# 0.0       0.92      0.99      0.95      1620
# 1.0       0.31      0.06      0.09       143
# accuracy                           0.91      1763
# macro avg       0.61      0.52      0.52      1763
# weighted avg       0.87      0.91      0.88      1763