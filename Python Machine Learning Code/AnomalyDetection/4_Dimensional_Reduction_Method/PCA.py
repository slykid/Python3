import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('qtagg')

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris.data, iris.target],
                  columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
df.head()

# 사용 모델 선언
scaler = StandardScaler()
pca = PCA()

# 파이프라인 생성
pipeline = make_pipeline(scaler, pca)
pipeline.fit(df.drop(['target'], axis=1))

# 차원축소 결과 주성분 확인
features = range(pca.n_components_)
feature_df = pd.DataFrame(data=features, columns=['pc_feature'])

# 차원축소 결과 설명력
variance_df = pd.DataFrame(data=pca.explained_variance_ratio_, columns=['variance'])
pc_feature_df = pd.concat([feature_df, variance_df], axis=1)
pc_feature_df  # 주성분1, 2 만으로도 95% 정도 설명 가능

# 시각화
x = df.drop(['target'], axis=1).reset_index(drop=True)
y = df['target'].reset_index(drop=True).astype(str)

_X = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
pc = pca.fit_transform(_X)

pc_df = pd.DataFrame(pc, columns=['PC1', 'PC2']).reset_index(drop=True)
pc_df = pd.concat([pc_df, y], axis=1)

plt.rcParams['figure.figsize'] = [10, 10]
sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue=y, legend='brief', s=50, linewidth=0.5)



