import pandas as pd
from io import StringIO

data = '''
a,b,c,d
1,2,3,4
5,6,,8
9,10,11,12
'''

df = pd.read_csv(StringIO(data))
df

# null count 확인
df.isnull().sum()

# 데이터
df1 = df
df1.dropna(axis=0)
df1.dropna(axis=1)

data2 = '''
a,b,c,d
1,2,3,
5,6,,
,,,
'''
df2 = pd.read_csv(StringIO(data2))
df2
df2.dropna(how='all')
df2.dropna(axis=1, how='all')

df1
df1.dropna(thresh=4)

df1
df1.dropna(subset=['c'])


from sklearn.impute import SimpleImputer

imr = SimpleImputer(strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

import pandas as pd
data = pd.DataFrame([
    [1, 'B', 'S', 13],
    [2, 'R', 'D', 9],
    [3, 'R', 'H', 9],
    [4, 'B', 'C', 3],
    [5, 'B', 'S', 12]
])

data.columns=["no", "color", "shape", "number"]
data

shape_order = {
    'S' : 1,
    'D' : 2,
    'C' : 3,
    'H' : 4
}
data['shape'] = data['shape'].map(shape_order)
data

inv_shape_order = {v: k for k, v in shape_order.items()}
data['shape'].map(inv_shape_order)

import numpy as np
class_mapping = {label : idx for idx, label in enumerate(np.unique(data["shape"]))}
class_mapping

data["shape"] = data["shape"].map(class_mapping)
data

inv_class_mapping = {v : k for k, v in class_mapping.items()}
data["shape"] = data["shape"].map(inv_class_mapping)
data

from sklearn.preprocessing import LabelEncoder

labeler = LabelEncoder()
y = labeler.fit_transform(data["shape"].values)
y

labeler.inverse_transform(y)

y_color = labeler.fit_transform(data["color"].values)
y_color

import numpy as np
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(data[["shape"]].values)
encoder.categories_
df_dummy = pd.DataFrame(encoder.transform(data[["shape"]].values).toarray(), columns=["is_C", "is_D", "is_H", "is_S"])

data_prep = pd.concat([data, df_dummy], axis=1)
data_prep

pd.get_dummies(data["shape"])

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

df_wine = pd.DataFrame(load_wine().data, columns=load_wine().feature_names)
df_wine_label = pd.DataFrame(load_wine().target, columns=["class"])
df_wine = pd.concat([df_wine, df_wine_label], axis=1)
print(load_wine().DESCR)
print('레이블: ', np.unique(df_wine["class"]))

df_x, df_y = df_wine.iloc[:, 0:12].values, df_wine.iloc[:, 13].values
np.unique(df_y)
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=0, stratify=df_y)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
x_train_norm = mms.fit_transform(x_train)
x_test_norm = mms.fit_transform(x_test)



