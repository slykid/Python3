import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

# 데이터셋 로드 및 스케일링
iris = load_iris()
scaler = StandardScaler()

x = scaler.fit_transform(iris.data)
y = (iris.target == 2).astype(np.float64)  # Y = 1 / N = 0

# Stratified K-Fold 설정 (계층적 교차 검증)
skf = StratifiedKFold(n_splits=10)

predictions = []
train_count = 0

# 모델 정의
model = SVC()

# 계층적 교차 검증 수행
for train_idx, test_idx in skf.split(x, y):
    train_count += 1

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 모델 학습
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # 정확도 계산
    total = len(y_test)
    correct = np.sum(y_pred == y_test)

    # 각 fold 결과 출력
    accuracy = round(100 * correct / total, 2)
    print("{} 회 학습 결과 : ".format(train_count), accuracy, "%")
    predictions.append(accuracy)

# 전체 교차 검증 결과 출력
print("="*10)
print("{}-Folds 학습 결과 : ".format(train_count), round(np.mean(predictions), 2), "%")
