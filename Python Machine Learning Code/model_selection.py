import numpy as np

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                    test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
print(mae)

mse = mean_squared_error(y_pred=y_pred, y_true=y_test)
print(mse)

rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))
print(rmse)

r2 = r2_score(y_pred=y_pred, y_true=y_test)
print(r2)

#=====================================================================
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.model_selection import KFold

iris = load_iris()
scaler = StandardScaler()

x = scaler.fit_transform(iris.data)
y = (iris.target == 2).astype(np.float64) # Y = 1 / N = 0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = SVC(kernel='linear', C=10**9)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

kfold = KFold(n_splits=10)

predictions = []
train_count = 0
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
print("{}-Fold 학습 결과 : ".format(train_count), round(np.mean(predictions), 2), "%")

