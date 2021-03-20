import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

# 1. 데이터 로드
train = pd.read_csv("Kaggle/2_Poker_Rule_Induction/data/train.csv")
test = pd.read_csv("Kaggle/2_Poker_Rule_Induction/data/test.csv")

## 1) 데이터 확인
train.head()
test.head()

train.info()
test.info()

# train.describe()

## 2) Data split - 학습용 데이터 : 검증용 데이터 = 7 : 3
train_index = [x for x in train.index % 3 != 0]
valid_index = [x for x in train.index % 3 == 0]


# 2. 더미변수 생성
train1 = pd.concat([
    pd.get_dummies(train.S1),
    pd.get_dummies(train.S2),
    pd.get_dummies(train.S3),
    pd.get_dummies(train.S4),
    pd.get_dummies(train.S5)], axis=1)

train1 = pd.concat([train, train1],axis=1)
train1 = train1.drop(['S1','S2','S3','S4','S5'],axis=1)
train1.info()

test1 = pd.concat([
    pd.get_dummies(test.S1),
    pd.get_dummies(test.S2),
    pd.get_dummies(test.S3),
    pd.get_dummies(test.S4),
    pd.get_dummies(test.S5)], axis=1)

test1 = pd.concat([test, test1],axis=1)
test1 = test1.drop(['S1','S2','S3','S4','S5'],axis=1)
test1.info()

# 3. Modeling #1. kNN
## 1) make data set
x_train = train1.drop("hand", axis=1)
y_train = train1["hand"]
x_test = test1.drop("id", axis=1)
id = test1["id"]

## 2) modeling
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

## 3) make prediction & submission
y_pred = model.predict(x_test)

submission1 = pd.DataFrame({'id':id, 'hand':y_pred})
submission1 = submission1[['id','hand']]
submission1.to_csv('Kaggle/2_Poker_Rule_Induction/result/sub.csv',index=False)

# 4. 성능개선
## 1) 데이터 전처리 : 문양-숫자 결합 후 학습
for suit, rank in zip(train['S1'],train['C1']):
    train[str(suit)+str(rank)] = np.where((train['S1']==suit) & (train['C1'] == rank), 1, 0)
for suit, rank in zip(train['S2'],train['C2']):
    train[str(suit)+str(rank)] = np.where((train['S2']==suit) & (train['C2'] == rank), 1, 0)
for suit, rank in zip(train['S3'],train['C3']):
    train[str(suit)+str(rank)] = np.where((train['S3']==suit) & (train['C3'] == rank), 1, 0)
for suit, rank in zip(train['S4'],train['C4']):
    train[str(suit)+str(rank)] = np.where((train['S4']==suit) & (train['C4'] == rank), 1, 0)
for suit, rank in zip(train['S5'],train['C5']):
    train[str(suit)+str(rank)] = np.where((train['S5']==suit) & (train['C5'] == rank), 1, 0)

for suit, rank in zip(test['S1'],test['C1']):
    test[str(suit)+str(rank)] = np.where((test['S1']==suit) & (test['C1'] == rank), 1, 0)
for suit, rank in zip(test['S2'],test['C2']):
    test[str(suit)+str(rank)] = np.where((test['S2']==suit) & (test['C2'] == rank), 1, 0)
for suit, rank in zip(test['S3'],test['C3']):
    test[str(suit)+str(rank)] = np.where((test['S3']==suit) & (test['C3'] == rank), 1, 0)
for suit, rank in zip(test['S4'],test['C4']):
    test[str(suit)+str(rank)] = np.where((test['S4']==suit) & (test['C4'] == rank), 1, 0)
for suit, rank in zip(test['S5'],test['C5']):
    test[str(suit)+str(rank)] = np.where((test['S5']==suit) & (test['C5'] == rank), 1, 0)

train2 = train.drop(['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5'],axis=1)
test2 = test.drop(['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5'],axis=1)

xtrain = train2.drop('hand',axis=1)
ytrain = train2['hand']
id = test2['id']
xtest = test2.drop('id',axis=1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
y_pred2 = model.predict(x_test)

submission2 = pd.DataFrame({'id': id, 'hand': y_pred2})
submission2 = submission2[['id', 'hand']]
submission2.to_csv('Kaggle/2_Poker_Rule_Induction/result/sub2.csv', index=False)

## 2) 문양 숫자 결합한 후 더미화
## - 문양과 숫자가 모두 같은 경우에만 1, 나머지는 0
train['SC1'] = train['S1'].apply(lambda x: str(x)) + train['C1'].apply(lambda x: str(x))
train['SC2'] = train['S2'].apply(lambda x: str(x)) + train['C2'].apply(lambda x: str(x))
train['SC3'] = train['S3'].apply(lambda x: str(x)) + train['C3'].apply(lambda x: str(x))
train['SC4'] = train['S4'].apply(lambda x: str(x)) + train['C4'].apply(lambda x: str(x))
train['SC5'] = train['S5'].apply(lambda x: str(x)) + train['C5'].apply(lambda x: str(x))
x_train = pd.get_dummies(train['SC1']) + pd.get_dummies(train['SC2']) +pd.get_dummies(train['SC3']) + pd.get_dummies(train['SC4']) + pd.get_dummies(train['SC5'])

test['SC1'] = test['S1'].apply(lambda x: str(x)) + test['C1'].apply(lambda x: str(x))
test['SC2'] = test['S2'].apply(lambda x: str(x)) + test['C2'].apply(lambda x: str(x))
test['SC3'] = test['S3'].apply(lambda x: str(x)) + test['C3'].apply(lambda x: str(x))
test['SC4'] = test['S4'].apply(lambda x: str(x)) + test['C4'].apply(lambda x: str(x))
test['SC5'] = test['S5'].apply(lambda x: str(x)) + test['C5'].apply(lambda x: str(x))
x_test = pd.get_dummies(test['SC1']) + pd.get_dummies(test['SC2']) +pd.get_dummies(test['SC3']) + pd.get_dummies(test['SC4']) + pd.get_dummies(test['SC5'])

