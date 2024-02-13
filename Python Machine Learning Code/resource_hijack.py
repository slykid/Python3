import time
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score

# data load
data = pd.read_csv("data/security/youtube_dataset.csv")
print(data.info())

## Column rename
data.columns = ["timestamp", "cpu", "virtual_memory", "disk_use_pct", "read_cnt", "write_cnt", "read_byte", "write_byte",
              "rss", "vms", "thread_cnt", "page_fault_cnt", "context_switch_cnt", "normal"]

# Preprocessing
case1 = data[["cpu", "virtual_memory", "disk_use_pct", "read_cnt", "write_cnt", "read_byte", "write_byte",
              "rss", "vms", "thread_cnt", "page_fault_cnt", "context_switch_cnt", "normal"]]

## Split feature & label
feature = case1[["cpu", "virtual_memory", "disk_use_pct", "read_cnt", "write_cnt", "read_byte", "write_byte",
              "rss", "vms", "thread_cnt", "page_fault_cnt", "context_switch_cnt"]].to_numpy()

label = case1["normal"]

## Split train, test set
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, shuffle=True, stratify=label)

## Scaling
### train set
scaler = StandardScaler()
x_train_scaled = scaler.fit(x_train).fit_transform(x_train)

### test set
scaler = StandardScaler()
x_test_scaled = scaler.fit(x_test).fit_transform(x_test)

# Model 1: SVC (SVM Classifier)
model = SVC(gamma='auto', kernel='rbf', probability=True)

## train model
start_time = time.time()
model.fit(x_train_scaled, y_train)
end_time = time.time()

print("Model Training....{}".format(end_time - start_time) + " sec")

## make prediction
y_pred = model.predict(x_test_scaled)
print(y_pred)
print(f1_score(y_true=y_test, y_pred=y_pred))  # f1-score: 1.0  -> overfitting


# Model 2: RandomForest
model = RandomForestClassifier(n_estimators=200, criterion="entropy")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(precision_score(y_test, y_pred))
print(f1_score(y_test, y_pred))  # f1-score: 1.0  -> overfitting