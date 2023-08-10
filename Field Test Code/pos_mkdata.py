import pandas as pd
from kiwipiepy import Kiwi
from gensim.models import word2vec


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
data["keywords"] = data["keywords"].apply(lambda x: " ".join(x))
data = data[["id", "prod_nm", "keywords", "label", "length"]]
data.to_csv("data/pos_menu/pos_menu_kiwi.csv", index=False)

# 불용어 처리를 위한 단어간 군집화
## 토크나이징 결과 취합
tokens = list(set(sum([x.split(" ") for x in data["keywords"].tolist()], [])))
tokens = tokens.sort()

## skip gram 을 통한 임베딩
