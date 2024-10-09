# 문제 상황
# 냉장고를 생산하는 업체에서 주요 공정인 A 단계에 사용되는 베어링이 있는데, 사용할 수록 마모되어, 주기적으로 변경해야한다.
# 또한 하나의 공정이 문제가 되면 All stop 이 되는 컨베이어 벨트 방식으로 생산한다.
# 때문에 베어링이 파손되기 이전에 이상점을 감지해 사전에 유지보수를 하고 싶어한다.
import os
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

matplotlib.use("MacOSX")
plt.style.use(['dark_background'])

# _path = "/Volumes/LaCie/workspace/Python/Dataset/anomaly_detection/bearing_failure"
_path = "/Users/kilhyunkim/workspace/Python3/DataAnalysis/Dataset/anomaly_detection/bearing_failure/"

# 1. 원천 데이터 로드
if os.path.exists(_path + "/data.csv") is False:
    print("File doesn't exist!!")
    print("Making dataset!!")

    data = pd.DataFrame()

    for filename in os.listdir(os.path.join(_path, "raw")):
        dataset = pd.read_csv(os.path.join(_path, "raw", filename), sep="\t")
        dataset_mean_abs = np.array(dataset.abs().mean())
        dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, 4))
        dataset_mean_abs.index = [filename]

        data = pd.concat([data, dataset_mean_abs], axis=0) ## append() 메소드는 pandas 2.x 버전 이상부터 deprecated 처리됨 -> concat() 메소드로 대체

    data.columns = ["Bearing1", "Bearing2", "Bearing3", "Bearing4"]
    data.head()

    data.index = pd.to_datetime(data.index, format="%Y.%m.%d.%H.%M.%S")
    data = data.sort_index()
    data.to_csv(_path + "/data.csv", index=True)
else:
    print("File exists!!")
    data = pd.read_csv(_path + "data.csv")

    # time 컬럼에 대한 추가적인 인덱스 변환 방법
    data["Unnamed: 0"] = pd.to_datetime(data["Unnamed: 0"])
    data.set_index("Unnamed: 0", inplace=True)
    data.index.name = ''

data.head()

# 2. 문제해결 프로세스 정의
# 2.1 문제정의
# - 냉장고 공정 주요 설비 A의 고장 발생 시, Line All stop 리스크 존재
# - A 설비의 핵심 부품인 Bearing 마모에 따른 설비 고장현상임

# 2.2 기대효과
# - 설비고장을 사전에 대비, Line stop 을 방지함
# - Line stop 에 따른 점검시간 축소 및 비용 절감 가능
# - 계획 대비 생산량 달성

# 2.3 해결방법
# - 데이터 전처리 및 EDA
# - 시계열 데이터 특성 추출
# - 이상 탐지 모델링 수행

# 2.4 성과측정
# - 모델 활용 전/후의 Line stop 횟수 비교
# - 모델 활용 전/후 계획 대비 생산률 비교


# 3. 데이터 전처리 및 EDA
# 3.1 데이터 형태 확인
print("data shape:", data.shape)
print(data.info())

# 3.2 NULL 값 확인
print(data.isnull().sum())

# 3.3 기술통계 확인
print(data.describe())

# 3.4 EDA
data.head()
print("Min datetime:", data.index.min())
print("Max datetime:", data.index.max())

plt.figure(figsize=(24, 5))
plt.plot(data.index, data["Bearing1"], linestyle='-', color='red', label='1')
plt.plot(data.index, data["Bearing2"], linestyle='-', color='yellow', label='2')
plt.plot(data.index, data["Bearing3"], linestyle='-', color='green', label='3')
plt.plot(data.index, data["Bearing4"], linestyle='-', color='lightblue', label='4')
plt.legend()

# 2004.02.17 일 이전 시점 데이터 확인
before = data[:"2004-02-17 00:00:00"]
plt.figure(figsize=(24, 5))
plt.plot(before.index, before["Bearing1"], linestyle='-', color='red', label='1')
plt.plot(before.index, before["Bearing2"], linestyle='-', color='yellow', label='2')
plt.plot(before.index, before["Bearing3"], linestyle='-', color='green', label='3')
plt.plot(before.index, before["Bearing4"], linestyle='-', color='lightblue', label='4')
plt.legend()

# 2004.02.17 일 이후 시점 데이터 확인
after = data["2004-02-17 00:00:00":]
plt.figure(figsize=(24, 5))
plt.plot(after.index, after["Bearing1"], linestyle='-', color='red', label='1')
plt.plot(after.index, after["Bearing2"], linestyle='-', color='yellow', label='2')
plt.plot(after.index, after["Bearing3"], linestyle='-', color='green', label='3')
plt.plot(after.index, after["Bearing4"], linestyle='-', color='lightblue', label='4')
plt.legend()

sns.displot(data["Bearing1"])


# 4. 시계열 데이터 특성 추출
data["year"] = data.index.year
data["month"] = data.index.month
data["day"] = data.index.weekday
data["hour"] = data.index.hour
data["date"] = data.index.date

data.head()

# 4.1 Bearing 일별 분포 확인
n_col = 1
n_row = 4

fig, ax = plt.subplots(ncols=n_col, nrows=n_row, figsize=(20, n_row * 5))
for row, col in enumerate(data.columns[0:4]):
    sns.boxplot(x='date', y=col, data=data, ax=ax[int(row%n_row)])
# plt.savefig()
# * 도출점: 이상 시점이 실제 시계열 데이터에서 보다 일찍 발생했다!

# 4.2 lag(지연) 데이터 생성
# - 시계열 데이터에서 이전에 값을 고려할 때 lag(지연) 데이터를 활용함
# - 시계열 데이터의 특징 상 이전의 결과에 영향을 많이 받음. 때문에 모델 성능을 높이고, 정교하게 만들기 위해 과거의 데이터를 활용해 학습한다.

# 4.2.1 shift 명령어를 사용한 lag 데이터 생성
data["Bearing1_lag"] = data["Bearing1"].shift(1)  # 1행 아래로 이동함
data[["Bearing1", "Bearing1_lag"]]

# - NaN 처리
data["Bearing1_lag"] = data["Bearing1"].shift(1, fill_value=0)
data[["Bearing1", "Bearing1_lag"]]

# 4.2.2 shift(n) 을 사용한 지연기간 조정
data["Bearing1_lag1"] = data["Bearing1"].shift(1, fill_value=0)
data["Bearing1_lag2"] = data["Bearing1"].shift(2, fill_value=0)
data[["Bearing1", "Bearing1_lag1", "Bearing1_lag2"]]

# 4.2.3 이동평균(Rolling Window)
# - smoothing 효과
data["bearing1_ma_3"] = data["Bearing1"].rolling(window=3).mean()
data[["Bearing1", "bearing1_ma_3"]]

# - NaN 처리
data["bearing1_ma_3"] = data["Bearing1"].rolling(window=3).mean()
data["bearing1_ma_3"] = data["bearing1_ma_3"].fillna(value=data["Bearing1"])  # inplace=True 방식은 pandas 3.0 에서 deprecated 예정
data[["Bearing1", "bearing1_ma_3"]]


# 5. 이상탐지 모델링
# 5.1 Model Selection
# - 선택 모델: PCA
# - 근거: 정상 데이터인 경우, 평균을 중심으로 데이터가 유동하고 있으며, 적절한 threshold 값이 설정되면, 이상치를 분류할 수 있음
data = data[["Bearing1", "Bearing2", "Bearing3", "Bearing4"]]

# scaler 생성
scaler = StandardScaler()

# 모델 생성
model = PCA()

# 파이프라인 생성
pipeline = make_pipeline(scaler, model)
pipeline.fit(data)

# 차원축소 주성분 개수 확인
features = range(model.n_components_)
df_features = pd.DataFrame(features, columns=["pc_feature"])
df_features.head()

# 모델 설명력 확인
df_valiance = pd.DataFrame(model.explained_variance_ratio_, columns=["variance"])
df_pca_features = pd.concat([df_features, df_valiance], axis=1)
print(df_pca_features)

#    pc_feature  variance
# 0           0  0.935243
# 1           1  0.052920
# 2           2  0.007789
# 3           3  0.004048
# -> 주성분 1, 2 만 사용해도 전체 분산의 98% 이상을 설명할 수 있음

# 6. 시각화 및 Threshold 설정
X_ = StandardScaler().fit_transform(data)

pca = PCA(n_components=2)
pc = pca.fit_transform(X_)

df_pc = pd.DataFrame(pc, columns=["PC1", "PC2"]).reset_index(drop=True)

plt.rcParams["figure.figsize"] = [5, 8]
sns.scatterplot(data=df_pc, x="PC1", y="PC2", legend="brief", s=50, linewidth=0.5)
plt.show()
# 각 클러스터 중심으로부터 멀리 떨어진 데이터(= 평균 보다 큰 데이터)일 수록 이상치로 판단가능!

# 6.1 Normal Gradle 설정 (-2 ~ 2 사이)
sns.scatterplot(data=df_pc, x="PC1", y="PC2", legend="brief", s=50, linewidth=0.5)
plt.vlines(-2, ymin=-1, ymax=1, color='r', linewidth=2)
plt.vlines(2, ymin=-1, ymax=1, color='r', linewidth=2)

plt.hlines(-1, xmin=-2, xmax=2, color='r', linewidth=2)
plt.hlines(1, xmin=-2, xmax=2, color='r', linewidth=2)

plt.gcf().set_size_inches(10, 10)

# 6.2 Abnormal Labeling
# -2 < PC1 < 2 && -1 < PC2 < 1 => 정상 (0) / 그 외 => 이상치 (1)
df_pc["abnormal"] = np.where( (df_pc["PC1"] > -2) & (df_pc["PC1"] < 2) & (df_pc["PC2"] > -1) & (df_pc["PC2"] < 1), 0, 1)
df_pc.head(5)
df_pc["abnormal"].value_counts()

# 6.3 기존 데이터와 비교
df_pc.index = data.index
df_pc.head(5)

df_result = pd.concat([data, df_pc], axis=1)
df_result.head()

# 6.4 Abnormal Points plot
df_abnormal = df_result[df_result["abnormal"] == 1]
df_normal = df_result[df_result["abnormal"] == 0]

for v, i in enumerate(data.columns[0:4]):
    plt.figure(figsize=(24, 15))
    plt.subplot(4, 1, v + 1)
    plt.plot(df_abnormal.index, df_abnormal[i], 'o', color='red', markersize=3)
    plt.plot(df_normal.index, df_normal[i], linestyle='--', color='grey')
    plt.title(i)
plt.show()