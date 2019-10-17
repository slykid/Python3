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


# 의사결정나무
import numpy as np
import os
import time

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap

from pydotplus import graph_from_dot_data

# 시각화 시 한글 깨짐에 대한 처리
font_location = 'C:/Windows/Fonts/NanumBarunGothic.ttf' # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

# 이미지 저장을 위한 경로 설정 및 폴더 생성
# PROJECT_ROOT_DIR = "D:\\workspace\\Python3"
PROJECT_ROOT_DIR = "D:\\workspace\\Python3"
CHAPTER_ID = "decision_trees"
if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is True:
    if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)) is True:
        time.sleep(1)
    else:
        os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))

elif os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is False:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images"))
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))
else:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

iris = datasets.load_iris()
x = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x, y)

export_graphviz(tree_clf, out_file=image_path("iris_tree.dot"),\
                feature_names=["꽃잎 길이 (cm)", "꽃잎 너비 (cm)"], class_names=iris.target_names,\
                rounded=True, filled=True)

import graphviz
with open("images\\decision_trees\\iris_tree.dot", 'rt', encoding='UTF8') as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='iris_tree', directory='images\\decision_trees', cleanup=True)
dot

# 엔트로피, 지니계수, 분류 오차에 대한 불순도 인덱스 비교
import matplotlib.pyplot as plt
import numpy as np

def gini(p):
    return (p) * (1 - p) + (1 - p) * (1- (1-p))

def entropy(p):
    return -p * np.log2(p) - (1 - p)*np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1-p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)

for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                           ['Entropy', 'Entropy(scaled)', 'Gini Impurity', 'Misclassification Error'],
                           ['-', '-', '--', '-.'],
                           ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle="--")
ax.axhline(y=1.0, linewidth=1, color='k', linestyle="--")
plt.ylim([0, 1.1])
plt.xlabel("p(i=1)")
plt.ylabel("Impurity Index")
plt.show()
save_fig("Compare Impurity Index with each label")

# 결정트리 만들기
iris = datasets.load_iris()

x = iris.data[:, [2, 3]]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z, alpha=.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = x[y == cl, 0], y = x[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx],
                    label=cl, edgecolors="black")

    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]

        plt.scatter(x_test[:, 0], x_test[:, 1],
                    c='', edgecolors="black", alpha=1.0,
                    linewidth=1, marker="o",
                    s=100, label="test set")

tree = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)
tree.fit(x_train, y_train)

x_combined = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(x_combined, y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel("Petal Length(cm)")
plt.ylabel("Petal Width(cm)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
save_fig("iris_decisionTreeClassifier")

dot_data = export_graphviz(tree, filled=True, rounded=True, class_names=["Setosa", "Versicolor", "Virginica"],
                           feature_names=["Petal Length", "Petal Width"], out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png("images/decision_trees/iris_decisionTreeDetail.png")

# 클래스 추정확률 계산
tree.predict_proba([[5, 1.5]])
tree.predict([[5, 1.5]])

# 의사 결정 나무 회귀
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# 추가적인 비교
# 2차식으로 만든 데이터셋 + 잡음
np.random.seed(42)
m = 200
x = np.random.rand(m, 1)
y = 4 * (x - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(x, y)
tree_reg2.fit(x, y)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_regression_predictions(tree_reg1, x, y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "깊이=0", fontsize=15)
plt.text(0.01, 0.2, "깊이=1", fontsize=13)
plt.text(0.65, 0.8, "깊이=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.subplot(122)
plot_regression_predictions(tree_reg2, x, y, ylabel=None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
plt.text(0.3, 0.5, "깊이=2", fontsize=13)
plt.title("max_depth=3", fontsize=14)

save_fig("tree_regression_plot")
plt.show()

dot_data = export_graphviz(tree_reg1, filled=True, rounded=True, out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png("images/decision_trees/iris_decisionTreeRegressorDetail_1.png")

dot_data = export_graphviz(tree_reg2, filled=True, rounded=True, out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png("images/decision_trees/iris_decisionTreeRegressorDetail_2.png")

# 사이킷런 투표기반 분류기
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_moons

from sklearn.metrics import accuracy_score

x, y = make_moons(n_samples=500, noise=0.30, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

log_clf = LogisticRegression(solver='liblinear', random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma='auto', random_state=42)

## 직접 투표 분류기
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting="hard"
)
voting_clf.fit(x_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

## 간접 투표 분류기
log_clf = LogisticRegression(solver='liblinear', random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma='auto', probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting="soft"
)
voting_clf.fit(x_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# 배깅 & 페이스팅
## 배깅
import os
import time
import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from sklearn.metrics import accuracy_score

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap

from pydotplus import graph_from_dot_data

# 시각화 시 한글 깨짐에 대한 처리
font_location = 'C:/Windows/Fonts/NanumBarunGothic.ttf' # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

# 이미지 저장을 위한 경로 설정 및 폴더 생성
# PROJECT_ROOT_DIR = "D:\\workspace\\Python3"
PROJECT_ROOT_DIR = "D:\\workspace\\Python3"
CHAPTER_ID = "ensenble"
if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is True:
    if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)) is True:
        time.sleep(1)
    else:
        os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))

elif os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is False:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images"))
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))
else:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

x, y = make_moons(n_samples=500, noise=0.30, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=500,
                            max_samples=100,
                            bootstrap=True,
                            n_jobs=1)
bag_clf.fit(x_train, y_train)
y_pred = bag_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(x_train, y_train)
y_pred_tree = tree_clf.predict(x_test)
print(accuracy_score(y_test, y_pred_tree))

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, x, y)
plt.title("결정 트리", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, x, y)
plt.title("배깅을 사용한 결정 트리", fontsize=14)
save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()


## 랜덤 포레스트
import numpy as np
import os
import time

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap

# 시각화 시 한글 깨짐에 대한 처리
font_location = 'C:/Windows/Fonts/NanumBarunGothic.ttf' # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

# 이미지 저장을 위한 경로 설정 및 폴더 생성
# PROJECT_ROOT_DIR = "D:\\workspace\\Python3"
PROJECT_ROOT_DIR = "D:\\workspace\\Python3"
CHAPTER_ID = "ensenble"
if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is True:
    if os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)) is True:
        time.sleep(1)
    else:
        os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))

elif os.path.isdir(os.path.join(PROJECT_ROOT_DIR, "images")) is False:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images"))
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))
else:
    os.mkdir(os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID))

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

iris = datasets.load_iris()

x = iris.data[:, [2, 3]]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

model_forest = RandomForestClassifier(criterion="gini", n_estimators=25, random_state=1, n_jobs=2)
model_forest.fit(x_train, y_train)
y_pred = model_forest.predict(x_test)
print(accuracy_score(y_test, y_pred))

def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z, alpha=.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = x[y == cl, 0], y = x[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx],
                    label=cl, edgecolors="black")

    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]

        plt.scatter(x_test[:, 0], x_test[:, 1],
                    c='', edgecolors="black", alpha=1.0,
                    linewidth=1, marker="o",
                    s=100, label="test set")

x_combined = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(x_combined, y_combined, classifier=model_forest, test_idx=range(105, 150))
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
save_fig("RandomForest model result using iris data")

## OOB
bag_clf = BaggingClassifier(RandomForestClassifier(),\
                            n_estimators=500,\
                            bootstrap=True,\
                            n_jobs=1,\
                            oob_score=True)
bag_clf.fit(x_train, y_train)
print(str(round(bag_clf.oob_score_, 4)*100) + "%")  # 94.64%

y_pred = bag_clf.predict(x_test)
accuracy_score(y_test, y_pred) #  1.0

bag_clf = BaggingClassifier(DecisionTreeClassifier(),\
                            n_estimators=500,\
                            bootstrap=True,\
                            n_jobs=1,\
                            oob_score=True)

bag_clf.fit(x_train, y_train)
print(str(round(bag_clf.oob_score_, 4)*100) + "%")  # 95.54%

y_pred = bag_clf.predict(x_test)
accuracy_score(y_test, y_pred) #  1.0

bag_clf.oob_decision_function_