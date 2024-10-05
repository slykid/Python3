import numpy as np
import pandas as pd
import matplotlib

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from ucimlrepo import fetch_ucirepo

matplotlib.use("MacOSX")

# 데이터 로드
wine = fetch_ucirepo(id=109)
x, y = wine.data.features, wine.data.targets

# 매타데이터 및 변수 정보 확인
print(wine.metadata)
print(wine.variables)

# 1. 데이터 전처리
# 1.1 7:3 으로 데이터 split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)

# 1.2 표준화 처리
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)


# 2. 공분산 계산
cov = np.cov(x_train_std.T)  # 공분산 행렬 계산
eigen_vals, eigen_vector = np.linalg.eig(cov)  # 고유값 분해 -> 고유 벡터, 고유값

print('고유값 \n%s' % eigen_vals)

total = sum(eigen_vals)
exp_val = [(i / total) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(exp_val)

plt.bar(range(1, 14), exp_val, alpha=0.5, align='center', label='Individual Explained Varience')
plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative Explained Varience')
plt.ylabel('Explained Varience Ratio')
plt.xlabel('Principal Component Index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('/Users/kilhyunkim/Pictures/Variance Sum to Component Index.png')
plt.show()


# 3. 고유 벡터 생성 및 투영
# 3.1 고유 벡터 - 고유값 쌍(튜플형식)으로 구성 후, 내림차순으로 정렬
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vector[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
print(eigen_pairs)

# 3.2 투영 행렬 W 생성
# 정렬된 것 중에서 가장 큰 고유값 2개와 쌍을 이루는 고유 벡터들을 선택한다. (전체 분산의 약 60% 정도를 사용할 것으로 예상됨)
# 투영 행렬은 13 x 2  형식의 리스트로 저장함
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print(w)

# 4. PCA 부분공간 산출하기
# X' = XW ( 부분공간 : X' / X : 원본 , W : 투영행렬 )
x_train_pca = x_train_std.dot(w)
print(x_train_pca)

# 변환된 훈련데이터의 시각화
colors = ['r', 'g', 'b']
markers = ['s', 'x', 'o']
y_train_classes = y_train["class"].values
for label, color, marker in zip(np.unique(y_train_classes), colors, markers):
    plt.scatter(x_train_pca[y_train_classes==label, 0],
                x_train_pca[y_train_classes==label, 1],
                c=color, label=label, marker=marker)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('/Users/kilhyunkim/Pictures/wine_dim_reduction_result.png')
plt.show()