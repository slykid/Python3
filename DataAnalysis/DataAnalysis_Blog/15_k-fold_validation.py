import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import KFold

iris = load_iris()
scaler = StandardScaler()

x = scaler.fit_transform(iris.data)
y = (iris.target == 2).astype(np.float64) # Y = 1 / N = 0
kfold = KFold(n_splits=10)

predictions = []
train_count = 0

# 모델 정의
model = SVC()

for train_idx, test_idx in kfold.split(x, y):
    train_count += 1

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    total = len(x_test)
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct+=1

    print("{} 회 학습 결과 : ".format(train_count), round(100 * correct / total , 2), "%")
    predictions.append(round(100 * correct / total , 2))

print("="*10)
print("{}-Folds 학습 결과 : ".format(train_count), round(np.mean(predictions), 2), "%")