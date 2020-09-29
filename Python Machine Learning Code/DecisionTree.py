# 1. Decision Tree
import os
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data[:, 2:]
y = iris.target

model = DecisionTreeClassifier(max_depth=2)
model.fit(x, y)

export_graphviz(
                model,
                out_file='images/decision_trees/iris_tree.dot',
                feature_names=iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True,
                filled=True
                )

path = os.getcwd()
command = "dot -Tpng " + \
          os.path.join(path, "images", "decision_trees") + \
          "\\iris_tree.dot -o " + \
          os.path.join(path, "images", "decision_trees") + \
          "\\iris_tree.png"
print(command)
os.system(command)

