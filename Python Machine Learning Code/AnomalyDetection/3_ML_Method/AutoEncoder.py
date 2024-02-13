import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import generate_data, evaluate_print

matplotlib.use("qtagg")

contamination = 0.1
n_train = 20000
n_test = 2000
n_features = 300

x_train, x_test, y_train, y_test = generate_data(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination, random_state=42)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

clf_name = 'AutoEncoder'
clf = AutoEncoder(hidden_neurons=[300, 100, 100, 300], epochs=10, contamination=contamination)
clf.fit(x_train)

y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_

y_test_pred = clf.predict(x_test)
y_test_scores = clf.decision_function(x_test)

print(y_train_pred[0:5], y_train_scores[0:5])
pd.Series(y_test_pred).value_counts()

# threshold 결정을 위한 Z score 계산 함수
def mod_z(col):
    med_col = col.median()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = 0.7413 * ((col - med_col) / med_abs_dev)
    return np.abs(mod_z)

score_series = pd.Series(y_test_scores)
mod_z_scores = mod_z(score_series)
sns.displot(mod_z_scores)

print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)
