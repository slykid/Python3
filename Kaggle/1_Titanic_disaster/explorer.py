import numpy as np
import pandas as pd

train = pd.read_csv("Kaggle/1_Titanic_disaster/data/train.csv")
test = pd.read_csv("Kaggle/1_Titanic_disaster/data/test.csv")

# 1. 데이터 확인
print(train.head())
print(test.head())

print(train.columns)
print(test.columns)

print(train.info())
print(test.info())

# 2. 전처리를 위한 데이터 통합
## 1) test 데이터에 Survived 컬럼 추가
test["Survived"] = ''
test["Survived"] = test["Survived"].replace('', np.nan, regex=True)

print(test.head())
print(test.tail())

## 2) 데이터 통합
print(test.columns)
test = test[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked']]
data = pd.concat([train, test], ignore_index=True)

print(data.head())
print(data.tail())

# 2. 데이터 전처리
## 1) Age 의 공백 값 -> 평균 값으로 대체
print(np.mean(data["Age"]))

### NaN 갯수확인
count = 0
for i in range(len(data["Age"])):
       if np.isnan(data["Age"][i]):
         count += 1

print(str(count)+"\n")

data["Age"] = data["Age"].replace(np.nan, np.round(np.mean(data["Age"]), 0))

### 수정 결과 확인
count = 0
for i in range(len(data["Age"])):
       if np.isnan(data["Age"][i]):
         count += 1

print(str(count)+"\n")

## 2) Embarked 컬럼의 NaN 값을 최빈값으로 변경한다.
### NaN의 값 갯수확인
print(pd.unique(data["Embarked"]))

from collections import Counter
print(Counter(data["Embarked"]))
print(Counter(data["Embarked"]).most_common(1))  # 최빈값 확인
print(Counter(data["Embarked"]).most_common(1)[0][0]) # 최빈값 산출
data["Embarked"] = data["Embarked"].replace(np.nan, Counter(data["Embarked"]).most_common(1)[0][0])

### 변경 확인
print(pd.unique(data["Embarked"]))

## 3) Fare 컬럼의 NaN 값을 0으로 치환
data["Fare"] = data["Fare"].replace(np.nan, 0)

### 변환 결과 확인
count = 0
for i in range(len(data["Fare"])):
    if np.isnan(data["Fare"][i]):
        count += 1

print(str(count) + "\n")


# 3. 모델링
## 1) 데이터 처리
data_tree = data
data_columns = data.columns

## 문자형 데이터를 수치형으로 변경
for i in range(len(data_tree["PassengerId"])):
    # Sex
    # 0 = male, 1 = female
    if data_tree["Sex"][i] == "male":
        data_tree["Sex"][i] = 0
    elif data_tree["Sex"][i] == "female":
        data_tree["Sex"][i] = 1

    # Embarked
    # 0 = C, 1 = Q, 2 = S
    if data_tree["Embarked"][i] == 'C':
        data_tree["Embarked"][i] = 0
    elif data_tree["Embarked"][i] == 'Q':
        data_tree["Embarked"][i] = 1
    else:
        data_tree["Embarked"][i] = 2

train_tree = data_tree[data_columns][0:890]
test_tree = data_tree[data_columns][891:]

test_tree = test_tree.reset_index()
del(test_tree["index"])
del(test_tree["Survived"])

## 2) 모델링
### 의사결정나무
from sklearn.tree import DecisionTreeClassifier

feature = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
label = ['Survived']

### 의사결정나무를 사용하기 위해선 학습에 사용되는 모든 값들이 수치형 데이터로 구성되어야함
### - 사용 컬럼: "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
model1 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=1)
model1.fit(train_tree[feature], train_tree[label])

Survived = model1.predict(test_tree[feature])

### 결과 파일 포맷을 위한 ID 생성
ID = pd.DataFrame(test["PassengerId"])


submission = pd.concat([ID, pd.DataFrame(Survived, columns=["Survived"])], axis=1)
submission["Survived"] = submission["Survived"].astype("int32")  # 제출형식은 int32 형임 (안 지킬 경우 0점 처리됨)
submission.head()

import os

if os.path.exists("Kaggle/1_Titanic_disaster/result") == False:
    os.mkdir("Kaggle/1_Titanic_disaster/result")

submission.to_csv("Kaggle/1_Titanic_disaster/result/submission1.csv", index=False) # Accuracy : 0.77511


# 4. 성능 개선
## 1) 탑승객의 가족 구성 수에 따른 탑승객 규모 변수 생성
data["FamilySize"] = data["Parch"] + data["SibSp"]

### Single : 1인 가족, 0으로 표시
### Small  : 2~4인 가족, 1로 표시
### Big    : 5인 이상 가족, 2로 표시
data["FamilySize"] = data["FamilySize"].map(lambda x : 0 if x == 1 else 1 if x >= 2 and x < 5 else 2)
print(data["FamilySize"].head())

## 2) 티켓 수에 따른 탑승객 규모 변수 생성
ticket_cnt = {}
ticket_kwd = np.unique(data["Ticket"])

for kwd in ticket_kwd:
    idx = np.where(kwd == data["Ticket"])[0]
    for i in idx:
        ticket_cnt[i] = {}
        ticket_cnt[i]['TicketSize'] = len(idx)

TicketCount = pd.DataFrame(ticket_cnt).T
TicketCount = TicketCount.sort_index(ascending=True)
data["TicketSize"] = TicketCount
del TicketCount

### Single : 1인 가족, 0으로 표시
### Small  : 2~4인 가족, 1로 표시
### Big    : 5인 이상 가족, 2로 표시
data["TicketSize"] = data["TicketSize"].map(lambda x : 0 if x == 1 else 1 if x >= 2 and x < 5 else 2)

## 3) 나이에 대한 그룹 변수 생성
## - kid(age < 13) : 0
## - Teenage(age >= 13 & age < 18) : 1
## - adult(age >= 18 & age < 60) : 2
## - elder(age >= 60) : 3
data["AgeClass"] = data["Age"].map(
    lambda x : 0 if x < 13
        else 1 if x >= 13 and x < 18
        else 2 if x >= 18 and x < 60
        else 3
)

## 4) 성별 및 직급인 단어 추출하기
import re

passenger_name = data["Name"]
name_kwd = []

for name in passenger_name:
    name_kwd.append(re.sub("^.*, (.*?)\\..*$", "\\1", name))

print(np.unique(name_kwd))

data["name_kwd"] = name_kwd
del name_kwd

### Mr = Mr : 0
### Mme, Mrs = Mrs : 1
### Mile, Ms, Lady, Dona = Miss : 2
### Master, Don, Rev, Dr, Major, Sir, Col, Capt, the Countess, Jonkheer = Rank : 3

data["name_kwd"] = data["name_kwd"].map(
    lambda x : 0 if x == "Mr"
        else 1 if x in ["Mme", "Mrs"]
        else 2 if x in ["Mile", "Ms", "Lady", "Dona"]
        else 3
)

## 1) 데이터 처리
data_tree = data
data_columns = data.columns

## 문자형 데이터를 수치형으로 변경
for i in range(len(data_tree["PassengerId"])):
    # Sex
    # 0 = male, 1 = female
    if data_tree["Sex"][i] == "male":
        data_tree["Sex"][i] = 0
    elif data_tree["Sex"][i] == "female":
        data_tree["Sex"][i] = 1

    # Embarked
    # 0 = C, 1 = Q, 2 = S
    if data_tree["Embarked"][i] == 'C':
        data_tree["Embarked"][i] = 0
    elif data_tree["Embarked"][i] == 'Q':
        data_tree["Embarked"][i] = 1
    else:
        data_tree["Embarked"][i] = 2

train_tree = data_tree[data_columns][0:890]
test_tree = data_tree[data_columns][891:]

test_tree = test_tree.reset_index()
del(test_tree["index"])
del(test_tree["Survived"])

feature = ['Pclass', 'Sex', 'Embarked', 'FamilySize', 'TicketSize', 'AgeClass', 'name_kwd']
label = ['Survived']

model1 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=1)
model1.fit(train_tree[feature], train_tree[label])

Survived = model1.predict(test_tree[feature])

### 결과 파일 포맷을 위한 ID 생성
ID = pd.DataFrame(test["PassengerId"])

submission2 = pd.concat([ID, pd.DataFrame(Survived, columns=["Survived"])], axis=1)
submission2["Survived"] = submission2["Survived"].astype("int32")  # 제출형식은 int32 형임 (안 지킬 경우 0점 처리됨)
submission2.head()

import os

if os.path.exists("Kaggle/1_Titanic_disaster/result") == False:
    os.mkdir("Kaggle/1_Titanic_disaster/result")

submission2.to_csv("Kaggle/1_Titanic_disaster/result/submission2.csv", index=False) # Accuracy : 0.77511


# 5. 모델 변경
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    max_depth=4,
    random_state=1
)
model2.fit(train_tree[feature], train_tree[label])

Survived = model2.predict(test_tree[feature])

### 결과 파일 포맷을 위한 ID 생성
ID = pd.DataFrame(test["PassengerId"])

submission3 = pd.concat([ID, pd.DataFrame(Survived, columns=["Survived"])], axis=1)
submission3["Survived"] = submission3["Survived"].astype("int32")  # 제출형식은 int32 형임 (안 지킬 경우 0점 처리됨)
submission3.head()

import os

if os.path.exists("Kaggle/1_Titanic_disaster/result") == False:
    os.mkdir("Kaggle/1_Titanic_disaster/result")

submission3.to_csv("Kaggle/1_Titanic_disaster/result/submission3.csv", index=False) # Accuracy : 0.78468

# 6. 모델 파라미터 수정
model3 = RandomForestClassifier(n_estimators=2000, criterion='gini', max_depth=4, random_state=1, n_jobs=4)
model3.fit(train_tree[feature], train_tree[label])

Survived = model3.predict(test_tree[feature])

### 결과 파일 포맷을 위한 ID 생성
ID = pd.DataFrame(test["PassengerId"])

submission4 = pd.concat([ID, pd.DataFrame(Survived, columns=["Survived"])], axis=1)
submission4["Survived"] = submission4["Survived"].astype("int32")  # 제출형식은 int32 형임 (안 지킬 경우 0점 처리됨)
submission4.head()

import os

if os.path.exists("Kaggle/1_Titanic_disaster/result") == False:
    os.mkdir("Kaggle/1_Titanic_disaster/result")

submission4.to_csv("Kaggle/1_Titanic_disaster/result/submission4.csv", index=False) # Accuracy : 0.78229
