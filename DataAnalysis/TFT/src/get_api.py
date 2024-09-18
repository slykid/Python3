# 작성일자: 2024.09.18
# 작성자: 김 길 현
# 참고자료: https://developer.riotgames.com/apis
import os
import requests
import json
import time
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd

# 진행상황 확인용
tqdm.pandas()

# API 및 Request 요청값 설정
api_key = "RGAPI-314fdcec-9309-4b82-a9f0-fd72ee3a4368"

request_header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36"
    , "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
    , "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8"
    , "Origin": "https://developer.riotgames.com"
    , "X-Riot-Token": api_key
}

base_url = "https://kr.api.riotgames.com/tft/"

if os.path.exists("Dataset/TFT/USERS") is False:
    os.makedirs("Dataset/TFT/USERS")

# UserList 추출
# - 상위 티어(마스터 이상)에 대해서만 사용 예정
for level in ["challenger", "grandmaster", "master"]:
    if os.path.exists(f"Dataset/TFT/USERS/df_userid_{level}.csv") is False:

        # User List 수집
        url = base_url + "league/v1/" + level
        data = requests.get(url=url, headers=request_header).json()
        globals()[f"df_{level}"] = pd.DataFrame(data["entries"])

        globals()[f"df_{level}"].to_csv(f"Dataset/TFT/USERS/df_userid_{level}.csv", index=False)

    else:
        globals()[f"df_{level}"] = pd.read_csv(f"Dataset/TFT/USERS/df_userid_{level}.csv")

# PUUID 추출
for level in ["challenger", "grandmaster", "master"]:
    if os.path.exists(f"Dataset/TFT/USERS/df_puuid_{level}.csv") is False:

        globals()[f"df_puuid_{level}"] = pd.DataFrame(columns=["summonerId", "puuid"])

        for summoner_id in globals()[f"df_{level}"]["summonerId"]:
            print(summoner_id, "\n")

            # User PUUID 추출
            url = base_url + "league/v1/entries/by-summoner/" + summoner_id
            data = requests.get(url, headers=request_header).json()
            globals()[f"df_puuid_{level}"] = pd.concat([globals()[f"df_puuid_{level}"], pd.DataFrame(data)[["summonerId", "puuid"]]], axis=0)

            time.sleep(10)

        globals()[f"df_puuid_{level}"].to_csv(f"Dataset/TFT/USERS/df_puuid_{level}.csv", index=False)
        print(f"{level} 티어의 PUUID 수집을 완료했습니다.")

    else:
        globals()[f"df_puuid_{level}"] = pd.read_csv(f"Dataset/TFT/USERS/df_puuid_{level}.csv")


# 증강체, 유닛, 아이템 한글화 작업
# 현 시즌: 12 ('24.09.18)
# 패치버전: 14.18b
datadragon_path = "Dataset/TFT/DataDragon/data.json"
with open(datadragon_path, 'r') as json_data:
    data_dragon = json.load(json_data)

items = pd.DataFrame(data_dragon["items"])
sets = pd.DataFrame(data_dragon["sets"]).loc['trait', '12']
units =