import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.info()
df_test.info()

for suit, rank in zip(df_train['S1'],df_train['C1']):
    df_train[str(suit)+str(rank)] = np.where((df_train['S1']==suit) & (df_train['C1'] == rank),1,0)
for suit, rank in zip(df_train['S2'],df_train['C2']):
    df_train[str(suit)+str(rank)] = np.where((df_train['S2']==suit) & (df_train['C2'] == rank),1,0)
for suit, rank in zip(df_train['S3'],df_train['C3']):
    df_train[str(suit)+str(rank)] = np.where((df_train['S3']==suit) & (df_train['C3'] == rank),1,0)
for suit, rank in zip(df_train['S4'],df_train['C4']):
    df_train[str(suit)+str(rank)] = np.where((df_train['S4']==suit) & (df_train['C4'] == rank),1,0)
for suit, rank in zip(df_train['S5'],df_train['C5']):
    df_train[str(suit)+str(rank)] = np.where((df_train['S5']==suit) & (df_train['C5'] == rank),1,0)


for suit, rank in zip(df_test['S1'],df_test['C1']):
    df_test[str(suit)+str(rank)] = np.where((df_test['S1']==suit) & (df_test['C1'] == rank),1,0)
for suit, rank in zip(df_test['S2'],df_test['C2']):
    df_test[str(suit)+str(rank)] = np.where((df_test['S2']==suit) & (df_test['C2'] == rank),1,0)
for suit, rank in zip(df_test['S3'],df_test['C3']):
    df_test[str(suit)+str(rank)] = np.where((df_test['S3']==suit) & (df_test['C3'] == rank),1,0)
for suit, rank in zip(df_test['S4'],df_test['C4']):
    df_test[str(suit)+str(rank)] = np.where((df_test['S4']==suit) & (df_test['C4'] == rank),1,0)
for suit, rank in zip(df_test['S5'],df_test['C5']):
    df_test[str(suit)+str(rank)] = np.where((df_test['S5']==suit) & (df_test['C5'] == rank),1,0)

# test_index = [x for x in df_train.index if x % 3 == 0] #30%는 testset으로 사용
# train_index = [x for x in df_train.index if x % 3 != 0] #70%가 trainingset으로 이용

df_train = df_train.drop(['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5'],axis=1)
df_test = df_test.drop(['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5'],axis=1)


xtrain = df_train.drop('hand',axis=1)
ytrain = df_train['hand']
id = df_test['id']
xtest = df_test.drop('id',axis=1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)


submission = pd.DataFrame({'id':id, 'hand':y_pred})
submission = submission[['id','hand']]
submission.to_csv('sub.csv',index=False)
