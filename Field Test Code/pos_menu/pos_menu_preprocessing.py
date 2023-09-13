import numpy as np
import pandas as pd

from kiwipiepy import Kiwi


# 명사 추출 함수 정의
def noun_extract(text):

    results = []
    result = kiwi.analyze(str(text))

    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and (pos.startswith("N") or pos.startswith("SL")):
            results.append(token)
    return results


# Kiwi 토크나이저 생성
kiwi = Kiwi()
kiwi.load_user_dictionary("data/pos_menu/user_menu_dict.txt")

data_raw = pd.read_parquet("data/pos_menu/pos_bill_aggr.parquet")
data = data_raw.copy()
data

# 키워드 생성
data["keywords"] = data["prod_nm"].apply(noun_extract)
data[["prod_nm", "keywords"]]
print(data.columns)

# keywords = sum(data["keywords"], [])
# keywords

data["keywords"] = data["keywords"].apply(lambda x: " ".join(x))
data["keywords"] = data["keywords"].astype(str)

data["price"] = data["tot_sale_amt"] / data["sale_qty"]

data = data[
    [
        "label",
        "sale_mm",
        "shop_cd",
        "upjong3_cd",
        "upjong3_nm",
        "prod_nm",
        "keywords",
        "price",
        "sale_qty",
        "tot_sale_amt",
        "menu1_nm",
        "menu2_nm",
        "menu3_nm",
        "etc",
    ]
]

# df_keyword = pd.DataFrame(keywords, columns=["keywords"])

data.to_csv("data/pos_menu/pos_menu_target.csv", index=False)
# df_keyword.to_csv("data/pos_menu/keywords.csv", index=False)
