import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from gensim.models import Word2Vec


def vectorize(docs_list, model):
    """
    args:
        docs_list: List of documents
        model: Gensim's Word Embedding model
    """

    feature = []

    for tokens in docs_list:
        zero_vector = np.zeros(model.vector_size)
        vector = []

        for token in tokens:
            if token in model.wv:
                try:
                    vector.append(model.wv[token])
                except KeyError:
                    continue
        if vector:
            vector = np.asarray(vector)
            avg_vec = vector.mean(axis=0)
            feature.append(avg_vec)
        else:
            feature.append(zero_vector)

    return feature


data_raw = pd.read_csv("data/pos_menu/pos_menu_target.csv")
data = data_raw.copy()
data

data["menu1_nm"] = data["menu1_nm"].replace("기타", np.nan)
data["menu2_nm"] = data["menu2_nm"].replace("기타", np.nan)
data["menu3_nm"] = data["menu3_nm"].replace("기타", np.nan)
data

upjong3_list = pd.unique(data["upjong3_cd"]).tolist()
menu1_list = pd.unique(data["menu1_nm"].dropna()).tolist()
menu2_list = pd.unique(data["menu2_nm"].dropna()).tolist()
menu3_list = pd.unique(data["menu3_nm"].dropna()).tolist()

upjong3_list.sort()
menu1_list.sort()
menu2_list.sort()
menu3_list.sort()

# menu1 기준
menu1_dict = {}
menu1_dict_rev = {}

for idx in range(len(menu1_list)):
    menu1_dict[idx] = menu1_list[idx]
    menu1_dict_rev[menu1_list[idx]] = idx

# menu2 기준
menu2_dict = {}
menu2_dict_rev = {}

for idx in range(len(menu2_list)):
    menu2_dict[idx] = menu2_list[idx]
    menu2_dict_rev[menu2_list[idx]] = idx

# menu3 기준
menu3_dict = {}
menu3_dict_rev = {}

for idx in range(len(menu3_list)):
    menu3_dict[idx] = menu3_list[idx]
    menu3_dict_rev[menu3_list[idx]] = idx

# 업종기준
upjong3_dict = {}
upjong3_dict_rev = {}

for idx in range(len(upjong3_list)):
    upjong3_dict[idx] = upjong3_list[idx]
    upjong3_dict_rev[upjong3_list[idx]] = idx

data["upjong3_emb"] = data["upjong3_cd"].apply(lambda x: upjong3_dict_rev.get(x))
data["menu1_emb"] = data["menu1_nm"].apply(lambda x: menu1_dict_rev.get(x))
data["menu2_emb"] = data["menu2_nm"].apply(lambda x: menu2_dict_rev.get(x))
data["menu3_emb"] = data["menu3_nm"].apply(lambda x: menu3_dict_rev.get(x))

data = data[
    [
        "label",
        "shop_cd",
        "upjong3_cd",
        "upjong3_nm",
        "upjong3_emb",
        "prod_nm",
        "keywords",
        "price",
        "sale_qty",
        "tot_sale_amt",
        "menu1_nm",
        "menu1_emb",
        "menu2_nm",
        "menu2_emb",
        "menu3_nm",
        "menu3_emb",
        "etc",
    ]
]
data

del upjong3_list, menu1_list, menu2_list, menu3_list, upjong3_dict, menu1_dict, menu2_dict, menu3_dict, upjong3_dict_rev, menu1_dict_rev, menu2_dict_rev, menu3_dict_rev

# 토큰 생성
data["keywords"] = data["keywords"].astype(str)
data["tokens"] = data["keywords"].apply(lambda x: x.lower().strip().split())

tokens = data["tokens"]

model = Word2Vec(
    sentences=tokens,
    window=max(data["tokens"].apply(lambda x: len(x))),
    min_count=1,
    sg=1,
)

vectorized_tokens = vectorize(data["tokens"], model)
vectorized_tokens

token_scaler = StandardScaler()
price_scaler = StandardScaler()

scaled_vertorize_tokens = token_scaler.fit_transform(vectorized_tokens)
scaled_price = price_scaler.fit_transform(
    data["price"].to_numpy().reshape(len(data["price"]), -1)
)

features = np.concatenate((scaled_vertorize_tokens, scaled_price), axis=1)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(features)

labels = dbscan.fit_predict(features)
labels

anomaly_label = -1
