from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

x = iris.data[:, [2, 3]]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

model_forest = RandomForestClassifier(criterion="gini", n_estimators=25, random_state=1, n_jobs=2, oob_score=True)
model_forest.fit(x_train, y_train)
y_pred = model_forest.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(1 - model_forest.oob_score_)
