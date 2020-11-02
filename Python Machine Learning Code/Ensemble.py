import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

data = df_wine[df_wine["Class label"] != 1]

# x = data[['Alcohol', 'Malic acid', 'Ash',
#             'Alcalinity of ash', 'Magnesium', 'Total phenols',
#             'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
#             'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
#             'Proline']]
x = data[['Alcohol', 'OD280/OD315 of diluted wines']].values
y = data[["Class label"]].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
model = BaggingClassifier(
            base_estimator=tree,
            n_estimators=500,
            bootstrap=True,
            bootstrap_features=False,
            max_samples=1,
            max_features=1,
            n_jobs=1,
            random_state=1
        )

tree = tree.fit(x_train, y_train)
y_train_tree = tree.predict(x_train)
y_test_tree = tree.predict(x_test)

acc_tree_train = accuracy_score(y_train, y_train_tree)
acc_tree_test = accuracy_score(y_test, y_test_tree)
print('의사결정나무 훈련정확도: %.3f 테스트정확도: %.3f' % (acc_tree_train, acc_tree_test))

model = model.fit(x_train, y_train)
y_train_bag = model.predict(x_train)
y_test_bag = model.predict(x_test)

acc_bag_train = accuracy_score(y_train, y_train_bag)
acc_bag_test = accuracy_score(y_test, y_test_bag)
print('배깅 훈련정확도: %.3f 테스트정확도: %.3f' % (acc_bag_train, acc_bag_test))