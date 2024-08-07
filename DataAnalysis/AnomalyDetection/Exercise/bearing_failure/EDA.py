# 문제 상황
# 냉장고를 생산하는 업체에서 주요 공정인 A 단계에 사용되는 베어링이 있는데, 사용할 수록 마모되어, 주기적으로 변경해야한다.
# 또한 하나의 공정이 문제가 되면 All stop 이 되는 컨베이어 벨트 방식으로 생산한다.
# 때문에 베어링이 파손되기 이전에 이상점을 감지해 사전에 유지보수를 하고 싶어한다.
import os
import numpy as np
import pandas as pd

_path = "/Volumes/LaCie/workspace/Python/Dataset/anomaly_detection/bearing_failure"
data = pd.DataFrame()

for filename in os.listdir(os.path.join(_path, "raw")):
    dataset = pd.read_csv(os.path.join(_path, "raw", filename), sep="\t")
    dataset_mean_abs = np.array(dataset.abs().mean())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, 4))
    dataset_mean_abs.index = [filename]

    data = data.append(dataset_mean_abs)

data.columns = ["Bearing1", "Bearing2", "Bearing3", "Bearing4"]
data.head()

data.index = pd.to_datetime(data.index, format="%Y.%m.%d.%H.%M.%S")
data = data.sort_index()
data.to_csv(_path + "/data.csv")
print("Data shape:", data.shape)
print(data.head())