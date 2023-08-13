import pandas as pd
from kiwipiepy import Kiwi
from gensim.models import Word2Vec

from matplotlib import pyplot as plt


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
keywords = sum(data["keywords"], [])

data["keywords"] = data["keywords"].apply(lambda x: " ".join(x))
data = data[["id", "prod_nm", "keywords", "label", "length"]]
df_keyword = pd.DataFrame(keywords, columns=["keyword"])

data.to_csv("data/pos_menu/pos_menu_kiwi.csv", index=False)
df_keyword.to_csv("data/pos_menu/pos_keyword.csv", index=False)

# 불용어 처리를 위한 단어간 군집화
data = pd.read_csv("data/pos_menu/pos_menu_kiwi.csv")
keyword = pd.read_csv("data/pos_menu/pos_keyword.csv")

## 토크나이징 결과 취합
tokens = tokens.sort()

## 단어별 분포 확인
plt.plot()


## skip gram 을 통한 임베딩
model = Word2Vec(sentences=tokens, window=3, min_count=1, sg=1, iter=1000)
model.init_sims(replace=True)

vectors = model.wv
vocabs = vectors.vocab.keys()
vector_list = [vectors[v] for v in vocabs]

## 테스트
print(vectors.similarity("연어", "생선"))
print(vectors.similarity("연어", "알람"))

