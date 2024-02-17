import numpy as np
import pandas as pd
from kiwipiepy import Kiwi

# 1. 토크나이징 with Kiwi
data = pd.read_csv("../data/pos_menu/org/target.csv")
data  # 848930 rows × 10 columns

kiwi = Kiwi()
kiwi.load_user_dictionary("../data/pos_menu/org/user_dictionary.csv")
kiwi.prepare()


def noun_extract(text):
    results = []
    result = kiwi.analyze(str(text))

    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and (pos.startswith("N") or pos.startswith("SL")):
            results.append(token)
    return results


data["keywords"] = data["prod_nm"].apply(noun_extract)
data[["prod_nm", "keywords"]]

keywords = sum(data["keywords"], [])
keywords

data["keywords"] = data["keywords"].apply(lambda x: " ".join(x))
data["keywords"] = data["keywords"].astype(str)
data[
    [
        "no",
        "label",
        "shop_cd",
        "upjong3_cd",
        "upjong3_nm",
        "prod_nm",
        "keywords",
        "menu1_nm",
        "menu2_nm",
        "menu3_nm",
    ]
]

data = data[
    [
        "no",
        "label",
        "shop_cd",
        "upjong3_cd",
        "upjong3_nm",
        "prod_nm",
        "keywords",
        "menu1_nm",
        "menu2_nm",
        "menu3_nm",
    ]
]
df_keyword = pd.DataFrame(keywords, columns=["keywords"])

data.to_csv("../data/pos_menu/result/data_edit.csv", index=False)
df_keyword.to_csv("../data/pos_menu/result/keywords.csv", index=False)
