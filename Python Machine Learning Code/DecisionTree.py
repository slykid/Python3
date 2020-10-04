# 1. Decision Tree
import os
import pandas as pd

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

# 데이터 로드
iris = load_iris()
# x = iris.data[:, 2:]
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 모델 생성
model = DecisionTreeClassifier(max_depth=2)
model.fit(x_train, y_train)

model.score(x_train, y_train)
model.score(x_test, y_test)  # 과적합

# 의사결정나무 시각화
tree.plot_tree(model)

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

# Pruning
# 사전 작업
data = load_breast_cancer()
print(data.DESCR)
print(data.feature_names)

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

model.score(x_train, y_train)
model.score(x_test, y_test)

export_graphviz(
                model,
                out_file='images/decision_trees/breast_cancer_tree.dot',
                feature_names=data.feature_names,
                class_names=data.target_names,
                rounded=True,
                filled=True
                )

path = os.getcwd()
command = "dot -Tpng " + \
          os.path.join(path, "images", "decision_trees") + \
          "\\breast_cancer_tree.dot -o " + \
          os.path.join(path, "images", "decision_trees") + \
          "\\breast_cancer_tree.png"
print(command)
os.system(command)

model.get_params()

## pre pruning
train_result = []
test_result = []
criterions = []
max_depths = []
min_leaves = []

insert_criterion = ["gini", "entropy"]
# max_depth = 3
list_min_leaves = [i for i in range(1, 10)]
for criterion in insert_criterion:
    for depth in range(2, 7):
        for min_leaf in list_min_leaves:
            # tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
            #                               min_samples_leaf=min_leaf, random_state=42)

            tree = DecisionTreeClassifier(criterion=criterion, max_depth=depth,
                                          min_samples_leaf=min_leaf, random_state=42)

            tree.fit(x_train, y_train)

            train_result.append(tree.score(x_train, y_train))
            test_result.append(tree.score(x_test, y_test))

            criterions.append(criterion)

            # max_depths.append(max_depth)
            max_depths.append(depth)

            min_leaves.append(min_leaf)

result = pd.DataFrame()
result["Criterion"] = criterions
result["Depth"] = max_depths
result["MinLeafSize"] = min_leaves
result["Train_Acc"] = train_result
result["Test_Acc"] = test_result

print(result)
# depth = 3 일 경우,
#  2      gini      3            3   0.969849  0.964912
# 11   entropy      3            3   0.979899  0.970760

preprune_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=3)
preprune_model.fit(x_train, y_train)
preprune_model.get_params()

preprune_model.score(x_test, y_test)

## post pruning
cost_path = model.cost_complexity_pruning_path(x_train, y_train)
print(cost_path)

### ccp_alpha : 비용 복잡도 파라미터(Cost Complexity Parameter) / ccp_alpha 보다 큰 값을 할당하면, 일부노드에 대해 가지치기 수행
ccp_alphas, impurities = cost_path.ccp_alphas, cost_path.impurities

### impurity-alpha
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

model_list = []
for ccp_alpha in ccp_alphas:
    model = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    model.fit(x_train, y_train)
    model_list.append(model)

print("Number of nodes in the last tree is : {} with ccp_alpha: {}".format(model_list[-1].tree_.node_count, ccp_alphas[-1]))

# node & depth - alpha
model_list = model_list[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in model_list]
depth = [clf.tree_.max_depth for clf in model_list]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

# acc-alpha
train_scores = [clf.score(x_train, y_train) for clf in model_list]
test_scores = [clf.score(x_test, y_test) for clf in model_list]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

## min_sample_leaf, max_depth 파라미터 값을 수정해서 과적합을 방지


