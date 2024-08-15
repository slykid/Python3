# 문제 상황
# 냉장고를 생산하는 업체에서 주요 공정인 A 단계에 사용되는 베어링이 있는데, 사용할 수록 마모되어, 주기적으로 변경해야한다.
# 또한 하나의 공정이 문제가 되면 All stop 이 되는 컨베이어 벨트 방식으로 생산한다.
# 때문에 베어링이 파손되기 이전에 이상점을 감지해 사전에 유지보수를 하고 싶어한다.
import os
import numpy as np
import pandas as pd

# _path = "/Volumes/LaCie/workspace/Python/Dataset/anomaly_detection/bearing_failure"
_path = "/Users/kilhyunkim/workspace/Python3/DataAnalysis/Dataset/anomaly_detection/bearing_failure/"

# 1. 원천 데이터 로드
if os.path.exists(_path) is False:
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
    data = pd.read_csv(_path + "data.csv", index_col="Unnamed: 0")

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

