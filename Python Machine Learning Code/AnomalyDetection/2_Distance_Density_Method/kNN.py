import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print, generate_data
from pyod.models.knn import KNN

matplotlib.use('qtagg')

contamination = 0.1  # 이상치 분포
n_train = 200
n_test = 100

# 샘플데이터 생성
X_train, X_test, y_train, y_test = generate_data(n_train=n_train, n_test=n_test, n_features=2, contamination=contamination, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# 모델 생성 및 학습
clf_name = 'KNN'
clf = KNN()
clf.fit(X_train)

# train score
y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_

print(y_train_pred)
print(y_train_scores)

# test score
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

print(y_test_pred)
print(y_test_scores)

# ROC Curve
print("On Training Data: ")
evaluate_print(clf_name, y_train, y_train_scores)

print("On Test Data: ")
evaluate_print(clf_name, y_test, y_test_scores)

# 시각화
visualize(clf_name=clf_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, y_train_pred=y_train_pred, y_test_pred=y_test_pred, show_figure=True, save_figure=True)
