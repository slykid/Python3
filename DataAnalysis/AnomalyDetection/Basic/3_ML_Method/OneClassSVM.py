import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('qtagg')

# 샘플 데이터 생성
rng = np.random.RandomState(50)

x = 0.3 * rng.randn(100, 2)
x_train = np.r_[x + 2, x - 2]
x_train = pd.DataFrame(x_train, columns=['x1', 'x2'])

x = 0.3 * rng.randn(20, 2)
x_test = np.r_[x + 2, x - 2]
x_test = pd.DataFrame(x_test, columns=['x1', 'x2'])

x_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
x_outliers = pd.DataFrame(x_outliers, columns=['x1', 'x2'])

# 샘플 데이터 시각화
plt.style.use(['dark_background'])

plt.rcParams['figure.figsize'] = [10, 10]

p1 = plt.scatter(x_train.x1, x_train.x2, c='white', s=20*2, edgecolor='k', label='train obs.')
p2 = plt.scatter(x_test.x1, x_test.x2, c='green', s=20*2, edgecolor='k', label='new regular obs.')
p3 = plt.scatter(x_outliers.x1, x_outliers.x2, c='red', s=20*2, edgecolor='k', label='new abnormal obs.')

plt.legend()
plt.gcf().set_size_inches(5,5)

plt.show()
plt.savefig('/Users/kilhyunkim/Pictures/OneClassSVM.jpg')
plt.close()

# 모델 학습 및 평가
## SVM 모델 파라미터 설명
## - kernel: 사용 커널함수 (linear, rbf)
## - gamma: 서포트벡터와의 거리 / 크면 가까이 있는 데이터, 작으면 멀리있는 데이터
## - nu: 초평면 밖에 있는(outlier) 데이터의 비율
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(x_train)
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)
y_pred_abnormal = clf.predict(x_outliers)

# Outliers 라벨 할당
x_outliers = x_outliers.assign(y=y_pred_abnormal)

p1 = plt.scatter(x_train.x1, x_train.x2, c='white', s=20*2, edgecolor='k', label='train obs.')
p2 = plt.scatter(x_outliers.loc[x_outliers.y == 1, ['x1']], x_outliers.loc[x_outliers.y == 1, ['x2']], c='green', s=20*2, edgecolor='k', label='detected regular obs.')
p3 = plt.scatter(x_outliers.loc[x_outliers.y == -1, ['x1']], x_outliers.loc[x_outliers.y == -1, ['x2']], c='red', s=20*2, edgecolor='k', label='detected outliers')

plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
plt.gcf().set_size_inches(10, 10)
plt.show()
plt.savefig(fname='/Users/kilhyunkim/Pictures/OneClassSVM_mod.jpg')
plt.close()

print("테스트 데이터셋 정확도: ", list(y_pred_test).count(1) / y_pred_test.shape[0])
print("이상치 데이터셋 정확도: ", list(y_pred_abnormal).count(-1) / y_pred_abnormal.shape[0])
