import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# Data Load
df = pd.read_csv('data/bicycle_train.csv')
df_test = pd.read_csv('data/bicycle_test.csv')

# Data Preprocessing
## train data
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = [x.month for x in df['datetime']]
df['hour'] = [x.hour for x in df['datetime']]
for col in ['hour','month','season', 'weather']:
    unique = np.unique(df[col])
    for col2 in unique:
        col3 = col + str(col2)
        df[col3] = np.where(df[col] == col2, 1, 0)
df = df.drop(['hour','month','season','weather'], axis=1)
xtrain = df.drop(['count','registered','casual','datetime'],axis=1)
ytrain = df['count']

## test data
df_test['datetime'] = pd.to_datetime(df_test['datetime'])
df_test['month'] = [x.month for x in df_test['datetime']]
df_test['hour'] = [x.hour for x in df_test['datetime']]
for col in ['hour','month','season', 'weather']:
    unique = np.unique(df_test[col])
    for col2 in unique:
        col3 = col + str(col2)
        df_test[col3] = np.where(df_test[col]==col2, 1, 0)
df_test = df_test.drop(['hour','month','season','weather'], axis=1)
xtest = df_test.drop(['datetime'],axis=1)

# Modeling
cv = KFold(n_splits=5, shuffle=True)
print(cv)

## model definition
model = RandomForestRegressor(n_estimators=1000)
reg_model = RandomForestRegressor(n_estimators=10)
reg_model1 = LassoCV(cv=10, n_alphas=1000)

## modeling
meta_feature = []
meta_feature_A = []
meta_feature_y = []
for train_index, test_index in cv.split(xtrain):

    # Data Split
    xtrain_train = xtrain.iloc[train_index]
    xtrain_test = xtrain.iloc[test_index]
    ytrain_train = ytrain.iloc[train_index]
    ytrain_test = ytrain.iloc[test_index]

    # Regressor Fit
    reg_model.fit(xtrain_train, ytrain_train)
    reg_model1.fit(xtrain_train, ytrain_train)

    # coefficient
    meta_feature.extend(reg_model.predict(xtrain_test))
    meta_feature_A.extend(reg_model1.predict(xtrain_test))
    meta_feature_y.extend(ytrain_test)
print("finished")

## Model Fit
model.fit(xtrain,ytrain);print("complete")

# Predict
y_pred = model.predict(xtest)
y_pred[y_pred<0] = 0

# Make Submission
submission = pd.DataFrame({'datetime': df_test['datetime'],
                           'count' : y_pred.astype('int')})
submission = submission[['datetime','count']]
print(submission)

submission.to_csv("submission.csv",index=False)