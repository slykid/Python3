# 작성일자: 2023.08.25
# 작성자: 김 길 현

import numpy as np
import pandas as pd

from kiwipiepy import Kiwi

data = pd.read_csv("data/pos_menu/pos_menu.csv")
label = pd.read_csv("data/pos_menu/label.csv")

data = pd.merge(data, label, how="left", left_on="label", right_on="num")
data = data[["id", "prod_nm", "word", "label", "length"]]

cnt = 0
for i in range(1, len(data) // 50000 + 2):
    res = data[cnt : cnt + 50000]
    res.to_csv(f"data/pos_menu/org/pos_menu_{cnt}_{cnt + 50000}.csv", index=False)

    cnt += 50000
