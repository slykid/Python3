import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples

from gensim.models import Word2Vec

from matplotlib import pyplot as plt


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


def mbkmeans_clusters(X, k, batch_size, print_silhouette_yn):
    km = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
    km.fit_predict(X)

    print(f"For n_cluster: {k}")
    print(f"Silhoutte coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia: {km.inertia_}")

    if print_silhouette_yn:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")

        silhouette_values = []

        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )

        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )

        for value in silhouette_values:
            print(
                f"Cluster {value[0]}: Size:{value[1]} | Avg:{value[2]:.2f} | Min:{value[3]:.2f} | Max:{value[4]:.2f}"
            )

    return km, km.labels_


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
        "sale_qty",
        "price",
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

# 토큰 생성
data["keywords"] = data["keywords"].astype(str)
data["tokens"] = data["keywords"].apply(lambda x: x.lower().strip().split())

# 단어 임베딩
tokens = data["tokens"]
model = Word2Vec(
    sentences=tokens,
    window=max(data["tokens"].apply(lambda x: len(x))),
    min_count=1,
    sg=1,
)

vectorized_tokens = vectorize(data["tokens"], model)
vectorized_tokens

clustering, cluster_label = mbkmeans_clusters(
    vectorized_tokens,
    k=len(pd.unique(data.menu1_emb)),
    batch_size=500,
    print_silhouette_yn=True,
)
indices = clustering.fit_predict(vectorized_tokens)

print(cluster_label)
print(len(pd.unique(cluster_label)))
print(indices)

distances = np.linalg.norm(
    vectorized_tokens - clustering.cluster_centers_[cluster_label], axis=1
)
threshold = np.percentile(distances, 95)
cluster = pd.DataFrame(indices, columns=["cluster"])
distance = pd.DataFrame(distances, columns=["distance"])

result = pd.concat([data, cluster, distance], axis=1)

result.to_csv("../result/pos_menu/20230902/result.csv", index=False)

outliers = result[result["distance"] > threshold][
    ["prod_nm", "menu1_nm", "cluster", "distance"]
]
outliers

outliers.to_csv("../result/pos_menu/20230902/outliers.csv", index=False)

# from sklearn.externals import joblib
import joblib

joblib.dump(clustering, "../result/pos_menu/20230902/minibatch_kmeans.pkl")

import pandas as pd

# from sklearn.externals import joblib
import joblib

result = pd.read_csv("../result/pos_menu/20230902/result.csv")
outliers = pd.read_csv("../result/pos_menu/20230902/outliers.csv")
kmeans = joblib.load("../result/pos_menu/20230902/minibatch_kmeans.pkl")

# 각 라벨별로 이상탐지
sorted(pd.unique(outliers["cluster"]))

for label in sorted(pd.unique(outliers["cluster"])):
    print("Label: " + str(label) + "\n")
    print(outliers[outliers["cluster"] == label])
    print("\n")
    print(sorted(pd.unique(outliers[outliers["cluster"] == label]["prod_nm"])))
    print("\n")

print(sorted(pd.unique(data["menu1_emb"])))

# 단어 임베딩
tokens = data[data["menu1_emb"] == 1]["tokens"]
model = Word2Vec(
    sentences=tokens,
    window=max(data["tokens"].apply(lambda x: len(x))),
    min_count=1,
    sg=1,
)

vectorized_tokens = vectorize(tokens, model)
vectorized_tokens

# 엘보우 기법 적용
# - 최적의 K 값 찾기
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

sse = []

for k in range(1, 27):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(vectorized_tokens)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 27), sse, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()

clustering, cluster_label = mbkmeans_clusters(
    vectorized_tokens, k=3, batch_size=500, print_silhouette_yn=True
)
indices = clustering.fit_predict(vectorized_tokens)
distances = np.linalg.norm(
    vectorized_tokens - clustering.cluster_centers_[cluster_label], axis=1
)
threshold = np.percentile(distances, 95)

print("Threshold: ", threshold)
print("Distances: ")
print(distances)

result = data[data["menu1_emb"] == 1]
result = result.reset_index(drop=True)
cluster = pd.DataFrame(indices, columns=["cluster"])
distance = pd.DataFrame(distances, columns=["distance"])

result = pd.concat([result, cluster, distance], axis=1)

outliers = result[result["distance"] > threshold][
    ["prod_nm", "menu1_nm", "cluster", "distance"]
]
outliers

# 모델변경 (KMeans → DBSCAN)
from sklearn.preprocessing import StandardScaler

data

tokens = data[data["menu1_emb"] == 1]["tokens"]
model = Word2Vec(
    sentences=tokens,
    window=max(data["tokens"].apply(lambda x: len(x))),
    min_count=1,
    sg=1,
)

vectorized_tokens = vectorize(tokens, model)
vectorized_tokens

scaler = StandardScaler()
vectorized_scaled_tokens = scaler.fit_transform(vectorized_tokens)
vectorized_scaled_tokens

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(vectorized_scaled_tokens)

labels = dbscan.fit_predict(vectorized_scaled_tokens)
labels

anomaly_label = -1
anomalies = np.where(labels == anomaly_label)[0].tolist()
prod_list = data[data["menu1_emb"] == 1]["prod_nm"]
prod_list = prod_list.reset_index(drop=True)
prod_list[anomalies].to_csv(
    "../result/pos_menu/20230902/outliers_dbscan.csv", index=False
)
