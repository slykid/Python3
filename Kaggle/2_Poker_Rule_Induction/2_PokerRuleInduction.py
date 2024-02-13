import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoCV

df_train = pd.read_csv('data/poker_rule_induction/train.csv')
df_test = pd.read_csv('data/poker_rule_induction/test.csv')

df_train.info()
df_test.info()

test_index = [x for x in df_train.index if x % 3 == 0] #30%는 testset으로 사용
train_index = [x for x in df_train.index if x % 3 != 0] #70%가 trainingset으로 이용

df_train = df_train = df_train.drop(['S1','S2','S3','S4','S5'],axis=1)
df_test = df_test.drop(['S1','S2','S3','S4','S5'],axis=1)

df_train1 = pd.get_dummies(df_train.C1) + pd.get_dummies(df_train.C2) + pd.get_dummies(df_train.C3) + pd.get_dummies(df_train.C4) + pd.get_dummies(df_train.C5)
df_test1 = pd.get_dummies(df_test.C1) + pd.get_dummies(df_test.C2) + pd.get_dummies(df_test.C3) + pd.get_dummies(df_test.C4) + pd.get_dummies(df_test.C5)

xtrain = df_train1
ytrain = df_train['hand']
id = df_test['id']
xtest = df_test1

cv = StratifiedKFold(n_splits=5)

regs = [GradientBoostingClassifier(n_estimators=10),
    GradientBoostingClassifier(n_estimators=50),
    RandomForestClassifier(n_estimators=10),
    RandomForestClassifier(n_estimators=50),
    ExtraTreesClassifier(n_estimators=10),
    ExtraTreesClassifier(n_estimators=50),
    KNeighborsClassifier(n_neighbors=1),
    KNeighborsClassifier(n_neighbors=5),
    LassoCV(cv=10,n_alphas=1000),
    LassoCV(cv=5, n_alphas=1000)]

meta_feature = pd.DataFrame(np.zeros(xtrain.shape[0]))

for train_index, test_index in cv.split(xtrain, ytrain):
    xtrain_train = xtrain.loc[train_index]
    xtrain_test = xtrain.loc[test_index]
    ytrain_train = ytrain.loc[train_index]
    ytrain_test = ytrain.loc[test_index]

    reg_num = 0
    for reg in regs:
        reg_name = str(reg_num) + str(reg.__class__).split('.')[-1].split("'")[0]
        reg.fit(xtrain_train, ytrain_train)
        meta_feature.loc[test_index, reg_name] = reg.predict(xtrain_test)
        reg_num += 1
meta_feature = meta_feature.drop(0,axis=1)

meta_feature_test = pd.DataFrame(np.zeros(xtest.shape[0]))
reg_num = 0
for reg in regs:
    reg_name = str(reg_num) + str(reg.__class__).split('.')[-1].split("'")[0]
    reg.fit(xtrain, ytrain)
    meta_feature_test[reg_name] = reg.predict(xtest)
    reg_num += 1
meta_feature_test = meta_feature_test.drop(0, axis=1)
print("finished")

stacker = GradientBoostingClassifier(n_estimators=10)
stacker.fit(meta_feature,ytrain)
print("complete")
y_pred = stacker.predict(meta_feature_test)

submission = pd.DataFrame({'id':id, 'hand':y_pred})
submission = submission[['id','hand']]
submission.to_csv('data/poker_rule_induction/sub.csv',index=False)