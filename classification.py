from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

datasets.get_data_home()

mnist = fetch_mldata('MNIST original')
mnist

x, y = mnist["data"], mnist["target"]
x.shape
y.shape

some_digit = x[26000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

y[26000]

x_train, x_test, y_train, y_test = x[:60000], x[:60000], y[:60000], y[:60000]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train_4 = (y_train == 4)
y_test_4 = (y_test == 4)

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(x_train, y_train_4)

sgd_clf.predict([some_digit])

cross_val_score(sgd_clf, x_train, y_train_4, cv=3, scoring="accuracy")
# array([0.93470326, 0.979     , 0.97984899])

class Never4Classifier(BaseEstimator):
    def fit(self, x, y=None):
        pass
    def predict(self, x):
        return np.zeros((len(x), 1), dtype=bool)

never_4_clf = Never4Classifier()
cross_val_score(never_4_clf, x_train, y_train_4, cv=3, scoring="accuracy")
# array([0.90215, 0.9028 , 0.90295])

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_4, cv=3)

confusion_matrix(y_train_4, y_train_pred)

precision_score(y_train_4, y_train_pred)
recall_score(y_train_4, y_train_pred)
f1_score(y_train_4, y_train_pred)

y_score = sgd_clf.decision_function([some_digit])
y_score

threshold = 0
y_some_digit_pred = (y_score > threshold)
y_some_digit_pred

threshold = 1000000
y_some_digit_pred = (y_score > threshold)
y_some_digit_pred

y_score = cross_val_predict(sgd_clf, x_train, y_train_4, cv=3, method="decision_function")
y_score

precisions, recalls, thresholds = precision_recall_curve(y_train_4, y_score)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="정밀도")
    plt.plot(thresholds, recalls[:-1], "g-", label="재현율")
    plt.xlabel("임계값")
    plt.legend(loc="center left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
