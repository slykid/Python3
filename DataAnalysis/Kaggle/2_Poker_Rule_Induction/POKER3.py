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

df_train['SC1'] = df_train['S1'].apply(lambda x: str(x)) + df_train['C1'].apply(lambda x: str(x))
df_train['SC2'] = df_train['S2'].apply(lambda x: str(x)) + df_train['C2'].apply(lambda x: str(x))
df_train['SC3'] = df_train['S3'].apply(lambda x: str(x)) + df_train['C3'].apply(lambda x: str(x))
df_train['SC4'] = df_train['S4'].apply(lambda x: str(x)) + df_train['C4'].apply(lambda x: str(x))
df_train['SC5'] = df_train['S5'].apply(lambda x: str(x)) + df_train['C5'].apply(lambda x: str(x))
xtrain = pd.get_dummies(df_train['SC1']) + pd.get_dummies(df_train['SC2']) +pd.get_dummies(df_train['SC3']) + pd.get_dummies(df_train['SC4']) + pd.get_dummies(df_train['SC5'])

df_test['SC1'] = df_test['S1'].apply(lambda x: str(x)) + df_test['C1'].apply(lambda x: str(x))
df_test['SC2'] = df_test['S2'].apply(lambda x: str(x)) + df_test['C2'].apply(lambda x: str(x))
df_test['SC3'] = df_test['S3'].apply(lambda x: str(x)) + df_test['C3'].apply(lambda x: str(x))
df_test['SC4'] = df_test['S4'].apply(lambda x: str(x)) + df_test['C4'].apply(lambda x: str(x))
df_test['SC5'] = df_test['S5'].apply(lambda x: str(x)) + df_test['C5'].apply(lambda x: str(x))
xtest = pd.get_dummies(df_test['SC1']) + pd.get_dummies(df_test['SC2']) +pd.get_dummies(df_test['SC3']) + pd.get_dummies(df_test['SC4']) + pd.get_dummies(df_test['SC5'])


# test_index = [x for x in df_train.index if x % 3 == 0] #30%는 testset으로 사용
# train_index = [x for x in df_train.index if x % 3 != 0] #70%가 trainingset으로 이용

df_train = df_train.drop(['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5'],axis=1)
df_test = df_test.drop(['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5'],axis=1)


xtrain = df_train.drop('hand',axis=1)
ytrain = df_train['hand']
id = df_test['id']
xtest = df_test.drop('id',axis=1)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)


submission = pd.DataFrame({'id':id, 'hand':y_pred})
submission = submission[['id','hand']]
submission.to_csv('sub.csv',index=False)
