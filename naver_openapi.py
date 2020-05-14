# 작성 일자 : 2020.04.28 (화)
# 작성자 : 김 길 현
# S020 관련 네이버 OPEN API - 검색 을 이용한 장소정보 확인
import os
import sys
import urllib.request
import json
import re
import pandas as pd
from operator import eq
import datetime
import time

# 사용자 정보
client_id = "W4MTiNOZaTlRKmizQYSq" #YOUR_CLIENT_ID
client_secret = "xIUKcSqDba" #YOUR_CLIENT_SECRET

# 데이터 로드
# data = pd.read.csv()

## 샘플데이터
data = pd.read_csv("data/scrapInput.csv", encoding="euc-kr")
data = data.rename({"STORE_NM" : "store_nm", "ADDR" : "addr", "UPJONG_DESC" : "upjong_desc"}, axis=1)


# 검색 리스트 생성
searchList = []
for i in range(0, len(data["store_nm"])):
    addr = ''
    if re.search("-", data["addr"][i]) is not None:
        if eq(data["addr"][i].split(",")[0].split("-")[1], '0'):
           addr = data["addr"][i].split(",")[0].split("-")[0]
        else :
            addr = data["addr"][i].split(",")[0]
    else :
        addr = data["addr"][i]

    searchList.append(data["store_nm"][i] + " " + addr)

# Naver OpenAPI
# 검색키워드 & URL
# url 형식
# "https://openapi.naver.com/v1/search/local.xml?query=검색어&display=10&start=1&sort=random"
# - query : 검색어
# - display : 한 번에 출력할 개수(기본값 : 10 / 최대 : 30)
# - start : 검색 시작 위치
# - sort : random(기본값, 유사도순) , comment(카페/블로그 리뷰 개수 순)

# 검색 횟수
# - request - response 한 횟수 만큼 증가
# - 1일 최대 25000건 한도
start_time = datetime.datetime.now()  # 시작시간
done_time = ""
if len(searchList) < 25000:
    for i in range(len(searchList)):
        query = searchList[i]

        encText = urllib.parse.quote(query)  # 검색어 인코딩
        url = "https://openapi.naver.com/v1/search/local.json?query=" + encText + "&display=1&sort=random" # json 결과 / 유사도 가장 높은 거 1개만
        # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과

        requestCnt = 0

        # REQUEST API
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)

        # RESPONSE API
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        # 결과 확인
        if(rescode==200):
            response_body = response.read()
            result = response_body.decode('utf-8')
            print(result)
        else:
            print("Error Code:" + rescode)

        # 결과 문자열 -> .json 형식으로 변환
        result_json = json.loads(result)  # json.loads : 문자열 -> .json 형식으로 변환하는 함수

        # .json 형식에서 실제 결과만 가져오기
        result_item = result_json["items"]

        # 결과값 확인
        if len(result_item) > 0:
            result_dict = result_item[0]
        else:
            result_dict = {"title": None, "category": None, "telephone": None, "address": None, "roadAddress": None,
                           "mapx": None, "mapy": None, "description": None, "link": None}

        DF = pd.DataFrame.from_dict(result_dict, orient='index').T
        if i == 0:
            resultDF = DF[["title", "category", "telephone", "address", "roadAddress", "mapx", "mapy", "description", "link"]]
        else:
            DF = DF[["title", "category", "telephone", "address", "roadAddress", "mapx", "mapy", "description", "link"]]
            resultDF = pd.concat((resultDF, DF), axis=0, ignore_index=True)

        if i % 10 == 0 and i != 0:
            time.sleep(5)

else :
    for i in range(25000):
        query = searchList[i]

        encText = urllib.parse.quote(query)  # 검색어 인코딩
        url = "https://openapi.naver.com/v1/search/local.json?query=" + encText + "&display=1&sort=random" # json 결과 / 유사도 가장 높은 거 1개만
        # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과

        requestCnt = 0

        # REQUEST API
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)

        # RESPONSE API
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        # 결과 확인
        if(rescode==200):
            response_body = response.read()
            result = response_body.decode('utf-8')
            print(result)
        else:
            print("Error Code:" + rescode)

        # 결과 문자열 -> .json 형식으로 변환
        result_json = json.loads(result)  # json.loads : 문자열 -> .json 형식으로 변환하는 함수

        # .json 형식에서 실제 결과만 가져오기
        result_item = result_json["items"]

        # 결과값 확인
        if len(result_item) > 0:
            result_dict = result_item[0]
        else :
            result_dict = {"title" : None, "category" : None, "telephone" : None, "address" : None,
                           "roadAddress" : None, "mapx" : None, "mapy" : None, "description" : None, "link" : None}

        DF = pd.DataFrame.from_dict(result_dict, orient='index').T
        if i == 0:
            resultDF = DF[["title", "category", "telephone", "address", "roadAddress", "mapx", "mapy", "description", "link"]]
        else:
            DF = DF[["title", "category", "telephone", "address", "roadAddress", "mapx", "mapy", "description", "link"]]
            resultDF = pd.concat((resultDF, DF), axis=0, ignore_index=True)

        if i % 10 == 0 and i != 0:
            time.sleep(5)


    done_time = datetime.datetime.now()

    while (done_time - start_time).days >= 1:
        time.sleep(3600)
        done_time = datetime.datetime.now()

    for i in range(25000, len(searchList)):
        query = searchList[i]

        encText = urllib.parse.quote(query)  # 검색어 인코딩
        url = "https://openapi.naver.com/v1/search/local.json?query=" + encText + "&display=1&sort=random"  # json 결과 / 유사도 가장 높은 거 1개만
        # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과

        requestCnt = 0

        # REQUEST API
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)

        # RESPONSE API
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        # 결과 확인
        if (rescode == 200):
            response_body = response.read()
            result = response_body.decode('utf-8')
            print(result)
        else:
            print("Error Code:" + rescode)

        # 결과 문자열 -> .json 형식으로 변환
        result_json = json.loads(result)  # json.loads : 문자열 -> .json 형식으로 변환하는 함수

        # .json 형식에서 실제 결과만 가져오기
        result_item = result_json["items"]

        # 결과값 확인
        if len(result_item) > 0:
            result_dict = result_item[0]
        else:
            result_dict = {"title": None, "category": None, "telephone": None, "address": None, "roadAddress": None,
                           "mapx": None, "mapy": None, "description": None, "link": None}

        DF = pd.DataFrame.from_dict(result_dict, orient='index').T
        if i == 0:
            resultDF = DF[
                ["title", "category", "telephone", "address", "roadAddress", "mapx", "mapy", "description",
                 "link"]]
        else:
            DF = DF[["title", "category", "telephone", "address", "roadAddress", "mapx", "mapy", "description",
                     "link"]]
            resultDF = pd.concat((resultDF, DF), axis=0, ignore_index=True)

        if i % 10 == 0 and i != 0:
            time.sleep(5)


for i in range(len(resultDF["title"])):
    if resultDF["title"][i] is not None:
        # title 문자열 중 html 태그 제거
        resultDF["title"][i] = re.sub("<b>|</b>|J&amp;", "", resultDF["title"][i])

if os.path.isdir("result/S020") is False:
    os.makedirs("result/S020")

# 결과 파일 생성
if datetime.datetime.now().month < 10:
    resultDF.to_csv("result/S020/naverapi_result_%d0%d.csv" % ((datetime.datetime.now().year), (datetime.datetime.now().month)), index=False, encoding="utf8")
else:
    resultDF.to_csv("result/S020/naverapi_result_%d%d.csv" % ((datetime.datetime.now().year), (datetime.datetime.now().month)), index=False, encoding="utf8")
