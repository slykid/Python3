import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

matplotlib.use('qtagg')

# 샘플 데이터 생성
X, _ = make_blobs(n_samples=10, n_features=2, centers=1, random_state=1)
print(X)

# 이상치 추가
X[0,0] = 10
X[0,1] = -10

# EllipticEnvelope 를 사용해 Outlier를 검출하기 위한 객체를 생성
outlier_detector = EllipticEnvelope(contamination=.2)

# EllipticEnvelope 객체를 생성한 데이터 학습
outlier_detector.fit(X)

# Outlier 검출
## +1 이면, boundary 안에 들어온 정상 데이터
## -1 이며, boundary 밖에 위치한 outlier로 간주
pred = outlier_detector.predict(X)
print(pred)

df = pd.DataFrame(X, columns=['col1', 'col2'])
df['outlier'] = pred
print(df)

# 시각화
plt.style.use(['dark_background'])
sns.scatterplot(x='col1', y='col2', hue='outlier', data=df)
plt.show()
plt.close()