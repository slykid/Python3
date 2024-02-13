import numpy as np
import pandas as pd

# 데이터 로드
data_raw = pd.read_csv("../../data/pos_menu/pos_menu.csv", sep=",")
data = data_raw.copy()
data

data["menu1_nm"] = data["menu1_nm"].replace("기타", np.nan)
data["menu1_nm"] = data["menu1_nm"].replace("x", np.nan)
data["menu2_nm"] = data["menu2_nm"].replace("기타", np.nan)
data["menu2_nm"] = data["menu2_nm"].replace("x", np.nan)
data["menu3_nm"] = data["menu3_nm"].replace("기타", np.nan)
data["menu4_nm"] = data["menu3_nm"].replace("x", np.nan)

train = data.loc[-data["menu3_nm"].isna()]
test = data.loc[data["menu3_nm"].isna()]

print(train.shape)
print(test.shape)

# label 정보 저장
train["label"] = 0
label = pd.unique(train.menu3_nm.apply(lambda x: str(x))).tolist()
num_class = len(label)
label.sort()

label = [x for x in label if x != "nan"]
label_dict = {word: num for num, word in enumerate(label)}
train["label"] = train["menu3_nm"].apply(lambda x: label_dict.get(x))

# 문자열 길이 컬럼 추가
train["length"], test["length"] = 0, 0
train["length"] = train["prod_nm"].apply(lambda x: len(str(x)))
test["length"] = test["prod_nm"].apply(lambda x: len(str(x)))

# ID 컬럼 추가
train["id"], test["id"] = 0, 0
train["id"] = [x for x in range(1, len(train.prod_nm) + 1)]
test["id"] = [x for x in range(1, len(test.prod_nm) + 1)]

# 컬럼순서 정리
train = train[["id", "prod_nm", "keywords", "label", "length"]]
test = test[["id", "prod_nm", "keywords", "length"]]

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

df_label = pd.concat(
    [
        pd.DataFrame(label_dict.keys(), columns=["word"]),
        pd.DataFrame(label_dict.values(), columns=["num"]),
    ],
    axis=1,
)

train.to_csv("../data/pos_menu/train/train.csv", index=False, header=True)
test.to_csv("../data/pos_menu/test/test.csv", index=False, header=True)
df_label.to_csv("../data/pos_menu/org/label.csv", index=False, header=True)
