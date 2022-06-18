# 초기코드 - python
import pandas as pd
import numpy as np

# 데이터 로드
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train.head()
test.head()


# 데이터 전처리(범주형 변수)
def male_label(x):
    if x.__eq__("male"):
        return 0
    elif x.__eq__("female"):
        return 1


def cp_label(x):
    if x.__eq__("typical angina"):
        return 1
    elif x.__eq__("atypical angina"):
        return 2
    elif x.__eq__("non-anginal pain"):
        return 3
    else:
        return 0


def thal_label(x):
    if x.__eq__("normal"):
        return 1
    elif x.__eq__("fixed defect"):
        return 2
    elif x.__eq__("reversable defect"):
        return 3
    else:
        return 0


train["sex_prep"] = train["sex"].apply(male_label)
train["cp_prep"] = train["cp"].apply(cp_label)
train["thal_prep"] = train["thal"].apply(thal_label)

test["sex_prep"] = test["sex"].apply(male_label)
test["cp_prep"] = test["cp"].apply(cp_label)
test["thal_prep"] = test["thal"].apply(thal_label)

# 데이터 전처리: 수치형
train["thalach_prep"] = (train["thalach"] - train["thalach"].mean()) / train["thalach"].std()
train["oldpeak_prep"] = (train["oldpeak"] - train["oldpeak"].mean()) / train["oldpeak"].std()
train["trestbps_prep"] = (train["trestbps"] - train["trestbps"].mean()) / train["trestbps"].std()
train["chol_prep"] = (train["chol"] - train["chol"].mean()) / train["chol"].std()

test["thalach_prep"] = (test["thalach"] - test["thalach"].mean()) / test["thalach"].std()
test["oldpeak_prep"] = (test["oldpeak"] - test["oldpeak"].mean()) / test["oldpeak"].std()
test["trestbps_prep"] = (test["trestbps"] - test["trestbps"].mean()) / test["trestbps"].std()
test["chol_prep"] = (test["chol"] - test["chol"].mean()) / test["chol"].std()

# 데이터 스플릿

from sklearn.model_selection import train_test_split

x = train[["sex_prep", "cp_prep", "thalach_prep", "exang", "oldpeak_prep", "slope", "ca", "thal_prep", "trestbps_prep", "chol_prep", "fbs", "restecg"]]
y = train[["target"]]

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)


# 모델링1: 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_valid)

cm = confusion_matrix(y_true = y_valid, y_pred = y_pred)
print(cm)
print(classification_report(y_valid, y_pred, target_names=["disease_y", "disease_n"]))

y_res = model.predict(test[["sex_prep", "cp_prep", "thalach_prep", "exang", "oldpeak_prep", "slope", "ca", "thal_prep", "trestbps_prep", "chol_prep", "fbs", "restecg"]])
df = pd.concat([test["id"],pd.DataFrame(y_res, columns=["target"])], axis=1)

# csv 파일 저장 예시 - python
df.to_csv('submission2.csv')



