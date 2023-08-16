import numpy as np
import pandas as pd
from kiwipiepy import Kiwi

from gensim.models import Word2Vec

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples


def noun_extract(text):
    results = []
    result = kiwi.analyze(str(text))

    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and pos.startswith("N") or pos.startswith("SL"):
            results.append(token)

    return results


# data load
data = pd.read_csv("data/pos_menu/pos_menu.csv")

kiwi = Kiwi()
data["keywords"] = data["prod_nm"].apply(noun_extract)
data["keywords"] = data["keywords"].astype(str)
keywords = data["keywords"].map(lambda x: x.lower().strip().split())
keywords = keywords.tolist()
keywords = [x for y in keywords for x in y]

data["keywords"] = data["keywords"].apply(lambda x: " ".join(x))
data = data[["id", "prod_nm", "keywords", "label", "length"]]
data

df_keywords = pd.DataFrame(keywords, columns=["keywords"])

data.to_csv("data/pos_menu/pos_menu_kiwi.csv", index=False)
df_keywords.to_csv("data/pos_menu/keyword.csv", index=False)

# 불용어 처리를 위한 단어간 군집화
data_raw = pd.read_csv("data/pos_menu/pos_menu_kiwi.csv")
data = data_raw.copy()
data

data["token"] = data["keywords"].astype(str)
data["token"] = data["token"].map(lambda x: x.lower().strip().split(" "))

tokens = data["token"].values
tokens

## skip gram 을 통한 임베딩
model = Word2Vec(sentences=tokens, window=3, min_count=1, sg=1)

vectors = model.wv
w2v_dict = vectors.key_to_index
w2v_dict


def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)

    return features


features = vectorize(tokens, model)
features[0]


def mbkmeans(x, k, batch_size, print_silhouette_yn):
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(x)

    print
