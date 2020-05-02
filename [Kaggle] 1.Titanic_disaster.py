# 작성일자 : 2020.01.29
# 작성자 : 김 길 현
# 참고자료 : https://www.kaggle.com/startupsci/titanic-data-science-solutions

import numpy as np
import pandas as pd
from operator import eq

train = pd.read_csv("data/titanic/train.csv")
test = pd.read_csv("data/titanic/test.csv")

train.head()
train.tail()

test.head()
test.tail()

train.info()
test.info()

#  성별(sex), 승선표(embarked) 숫자로 변환
# 성별 : male = 1, female = 2
# 승선표 : NaN = 1 C = 2, Q = 3, S = 4
count = 0
for i in range(0, len(train["PassengerId"])):
    if pd.isna(train["Embarked"][i]) is True:
        count += 1
# Na 갯수 2 / 891 개

for i in range(0, len(train["PassengerId"])):
    if eq("male", train["Sex"][i]):
        train["Sex"][i] = 1
    else :
        train["Sex"][i] = 2


    if eq("C", train["Embarked"][i]):
        train["Embarked"][i] = 2
    elif eq("Q", train["Embarked"][i]):
        train["Embarked"][i] = 3
    elif eq("S", train["Embarked"][i]):
        train["Embarked"][i] = 4
    else :
        train["Embarked"][i] = 1

np.unique(train["Sex"])
np.unique(train["Embarked"])

# 통계치 확인
print(train.describe())
train.describe(include=['0'])