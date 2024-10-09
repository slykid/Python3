# [문제 상황]
# A사는 반도체를 생산하는 글로벌 회사인데, 반도체 구성품 중 Wafer는 반도체 집적회로의 핵심 재료이다.
# A사는 반도체의 성능을 향상시키기 위해서 최근 Wafer 설계를 변경하고 제품을 생산 중에 있으며, 설계 변경 과정에서 불량 제품이 발생하고 있다.
# 이에 대해 이상탐지 모델링을 통해 이상인 Wafer를 사전에 검출하고자 한다.

import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

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




