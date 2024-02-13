import os
import numpy as np
import pandas as pd
import pandasql as ps

from kiwipiepy import Kiwi

from tokenizers import BertWordPieceTokenizer


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


# WordPiece Tokenizer용 사전 생성
data = pd.read_csv("data/pos_menu/pos_menu_target.csv")

# WordPiece Tokenizer
tokenizer = BertWordPieceTokenizer(
    vocab="Models/bert-kor-base/vocab.txt",
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,  # Must be False if cased model
    lowercase=True,
    wordpieces_prefix="##",
)

tokenizer.train(
    files="data/pos_menu/menu_corpus.csv",
    limit_alphabet=6000,
    vocab_size=22000,
    min_frequency=25,
)

if os.path.isdir("result/pos_menu/20230916") is False:
    os.makedirs("result/pos_menu/20230916")

tokenizer.save("result/pos_menu/20230916/vocab.txt", True)

data["tokens"] = data["edit_prod_nm"].apply(lambda x: tokenizer.encode(str(x)).tokens)
data["tokens"]

# 토큰 정보 수정
def dictionary_based_post_process(tokens, user_dict):

    i = 0
    new_tokens = []

    while i < len(tokens):
        if str(tokens[i]).replace("##", "") in user_dict["menu_nm"].values():
            new_tokens.append(tokens[i])

        elif (
            i < len(tokens) - 1
            and f"{str(tokens[i]).replace('##', '')}{str(tokens[i+1]).replace('##', '')}"
            in user_dict["menu_nm"].values()
        ):
            new_tokens.append(
                f"{str(tokens[i]).replace('##', '')}{str(tokens[i+1]).replace('##', '')}"
            )

            i += 1

        elif (
            i < len(tokens) - 1
            and f"{str(tokens[i]).replace('##', '')} {str(tokens[i+1]).replace('##', '')}"
            in user_dict["menu_nm"].values()
        ):
            new_tokens.append(
                f"{str(tokens[i]).replace('##', '')} {str(tokens[i+1]).replace('##', '')}"
            )
            i += 1

    return new_tokens


menu_dict = pd.read_csv("data/pos_menu/user_menu_dict.txt", header=None, sep="\t")
menu_dict.columns = ["menu_nm", "tag", "score"]
menu_dict = menu_dict[["menu_nm"]]
menu_dict = menu_dict.to_dict()

new_tokens = []

for token_list in data["tokens"]:
    new_tokens += dictionary_based_post_process(token_list, menu_dict)

new_tokens

vocab = pd.read_csv("../model/bert-kor-base/vocab.txt")  # <- 에러발생
vocab

new_vocab = pd.concat([new_tokens, vocab], axis=0)
new_vocab.to_csv("result/pos_menu/20230911/vocab.txt")
