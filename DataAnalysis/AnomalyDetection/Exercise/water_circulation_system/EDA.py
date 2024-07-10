# [시나리오]
# 1. 문제정의
# - 사업 확장으로 인한 관리포인트 증가에 따른 운영 및 유지보수 자원 부족
# - Water Circulation System 고장으로 인한 리스크 발생

# 2. 기대효과
# - 이상진단 시스템을 ㅗㅇ한 사전 유지보수를 통해 고장 발생으로 인한 리스크 감소

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

import pyod

# 전역변수 및 설정
matplotlib.use("MacOSX")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

# 데이터 로드
df = pd.read_csv("Dataset/anomaly_detection/water_circulation_system/chapter01_df.csv", sep=";")
df.head()

# 1. 데이터 전처리 및 EDA




