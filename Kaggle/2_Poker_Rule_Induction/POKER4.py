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


# test_index = [x for x in df_train.index if x % 3 == 0] #30%는 testset으로 사용
# train_index = [x for x in df_train.index if x % 3 != 0] #70%가 trainingset으로 이용

df_train = df_train = df_train.drop(['S1','S2','S3','S4','S5'],axis=1)
df_test = df_test.drop(['S1','S2','S3','S4','S5'],axis=1)

df_train1 = pd.get_dummies(df_train.C1) + pd.get_dummies(df_train.C2) + pd.get_dummies(df_train.C3) + pd.get_dummies(df_train.C4) + pd.get_dummies(df_train.C5)
df_test1 = pd.get_dummies(df_test.C1) + pd.get_dummies(df_test.C2) + pd.get_dummies(df_test.C3) + pd.get_dummies(df_test.C4) + pd.get_dummies(df_test.C5)

xtrain = df_train1
ytrain = df_train['hand']
id = df_test['id']
xtest = df_test1

model = KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)


submission = pd.DataFrame({'id':id, 'hand':y_pred})
submission = submission[['id','hand']]
submission.to_csv('sub.csv',index=False)