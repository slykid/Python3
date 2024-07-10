import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('qtagg')

rng = np.random.RandomState(42)

# 샘플 데이터 생성
x_train = 0.2 * rng.randn(1000, 2)
x_train = np.r_[x_train + 3, x_train]
x_train = pd.DataFrame(x_train, columns=['x1', 'x2'])

x_test = 0.2 * rng.randn(200, 2)
x_test = np.r_[x_test + 3, x_test]
x_test = pd.DataFrame(x_test, columns=['x1', 'x2'])

x_outliers = rng.uniform(low=-1, high=5, size=(50, 2))
x_outliers = pd.DataFrame(x_outliers, columns=['x1', 'x2'])

# 시각화
plt.style.use(['dark_background'])
plt.rcParams['figure.figsize'] = [10, 10]

p1 = plt.scatter(x_train.x1, x_train.x2, c='white', s=20*2, edgecolor='k', label='train observations')
p2 = plt.scatter(x_test.x1, x_test.x2, c='green', s=20*2, edgecolor='k', label='new regular obs.')
p3 = plt.scatter(x_outliers.x1, x_outliers.x2, c='red', s=20*2, edgecolor='k', label='new abnormal obs.')

plt.legend()
plt.show()

plt.savefig(fname='/Users/kilhyunkim/Pictures/IsolationForest.jpg')
plt.close()

clf = IsolationForest(max_samples=100, contamination=0.1, random_state=42)
clf.fit(x_train)
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)
y_pred_abnormal = clf.predict(x_outliers)

print(y_pred_abnormal)  # -1 : Outlier / 1: Normal

# Outlier Score: 점수가 낮을수록 Outlier
print(clf.decision_function(x_outliers))
## decision_function을 사용하는 이유
## - 모델 파라미터 상의 comtamination 값을 통해 Outlier Score 를 -1 ~ 1 사이 값으로 정렬했을 때, 양 끝단에서부터 각 5%(총 10%)를 Outlier로 정의할 수 있지만,
##   threshold를 다시 조정해서, 이 보다 낮거나 높은 데이터를 다시 재조정하기 위해 재라벨링을 하고자 사용함

# Normal Score: 점수가 높을수록 정상
print(clf.decision_function(x_test)[0:5])

# Outlier 라벨 재할당
x_outliers = x_outliers.assign(y=y_pred_abnormal)

# 시각화 재수행(변경 결과 반영)
plt.style.use(['dark_background'])
plt.rcParams['figure.figsize'] = [10, 10]

# 학습데이터
p1 = plt.scatter(x_train.x1, x_train.x2, c='white', s=20*2, edgecolor='k', label='train observations')

# 실제 이상치
p2 = plt.scatter(x_outliers.loc[x_outliers.y==1, ['x1']], x_outliers.loc[x_outliers.y==1, ['x2']], c='green', s=20*2, edgecolor='k', label='detected regular obs.')

# 이상치지만, 정상으로 예측한 경우
p3 = plt.scatter(x_outliers.loc[x_outliers.y==-1, ['x1']], x_outliers.loc[x_outliers.y==-1, ['x2']], c='red', s=20*2, edgecolor='k', label='detected outliers')

plt.legend()
plt.gcf().set_size_inches(10,10)

plt.show()
plt.savefig(fname='/Users/kilhyunkim/Pictures/IsolationForest_mod.jpg')  # 시각화를 통한 평가: 별도의 라벨링 없이 잘 분류함
plt.close()

# 성능평가
print("테스트 데이터의 정확도: ", list(y_pred_test).count(1) / y_pred_test.shape[0])
print("이상치 데이터의 정확도: ", list(y_pred_abnormal).count(-1) / y_pred_abnormal.shape[0])