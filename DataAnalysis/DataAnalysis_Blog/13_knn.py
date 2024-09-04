import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import scipy.stats as stats

# 1. 사용 데이터 로드
cancer_data = pd.read_csv("~/workspace/Python3/DataAnalysis/Dataset/breast_cancer/wdbc.data", header=None)
cancer = cancer_data.iloc[:, 0:12]
cancer.columns = ["id", "diagnosis", "radius", "texture", "perimeter", \
      "area", "smoothness", "compactness", "concavity", \
      "concave_points", "symmetry", "fractal_dimension"]

print(cancer['diagnosis'].value_counts())

# 2. 원본 전처리
cancer['diagnosis'] = cancer['diagnosis'].map({'B': 0, 'M': 1})

print(round(cancer['diagnosis'].value_counts(normalize=True) * 100, 1))
print(cancer.describe())

# 데이터셋 분리
cancer_data = cancer[["id", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]]
cancer_label = cancer[["diagnosis"]]
x_train, x_test, y_train, y_test = train_test_split(cancer_data, cancer_label, test_size=0.2, random_state=1234)

# 모델링
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 성능개선
# 1. 데이터 전처리 추가
scaler = StandardScaler()
cancer_z = scaler.fit_transform(cancer[['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']])
cancer_z = pd.DataFrame(cancer_z, columns=['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'])
cancer_z = pd.concat([cancer["id"], cancer_z], axis=1)

x_train, x_test, y_train, y_test = train_test_split(cancer_z, cancer_label, test_size=0.2, random_state=1234)

knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(confusion_matrix(y_test, y_pred))

# 2. 모델 파라미터 수정
for k in [1, 5, 11, 15, 21, 27]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'k={k}')
    print(conf_matrix)
