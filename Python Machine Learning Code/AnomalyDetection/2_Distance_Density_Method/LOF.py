import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

matplotlib.use('qtagg')

# 샘플 데이터 생성
df_train = pd.DataFrame([
    [2, 1, 3],
    [3, 2, 5],
    [3, 4, 7],
    [5, 5, 10],
    [7, 5, 12],
    [2, 5, 7],
    [8, 9, 13],
    [9, 10, 13],
    [10, 50, 12],
    [11, 45, 13],
    [6, 12, 12],
], columns=['hour', 'attendance', 'score'])

df_test = pd.DataFrame([
    [9, 2, 13],
    [6, 10, 12],
    [2, 4, 6]
], columns=['hour', 'attendance', 'score'])

print(df_train)
print(df_test)

# 모델 생성 및 예측
outlier = LocalOutlierFactor(n_neighbors=2, contamination=0.2)
y_pred = outlier.fit_predict(df_train)
df_train['outlier'] = y_pred

print(df_train)  # -1 : 이상치를 의미함

# LOF 값 확인
outlier.negative_outlier_factor_  # 학습 샘플 데이터 상의 LOF 점수


# 3D 시각화 생성
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
df_red = df_train[df_train['outlier'] == 1]
df_green = df_train[df_train['outlier'] == -1]

ax.scatter(df_red['hour'], df_red['attendance'], df_red['score'], color='r', alpha=0.5)
ax.scatter(df_green['hour'], df_green['attendance'], df_green['score'], color='g', alpha=0.5)

plt.show()
plt.savefig(fname='/Users/kilhyunkim/Pictures/LOF_3D.jpg')
plt.close()