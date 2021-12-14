import requests, bs4
import pandas as pd
import time

# url 파라미터
# - numOfRows 값은 100으로 설정함
page_no = 1

# 데이터 저장 변수 선언
content = {}
content_count = 0

while True:
    # 수집
    print(str(page_no) + "\n")
    url = f"http://apis.data.go.kr/1613000/AptListService2/getTotalAptList?serviceKey=n1tWqYKnI0IwXCLBdC8U3oGZc1w5Tb9KsDh2vPyZ44Sl5OyMmv%2FDMCk%2BeUgWWSGFWWxFbQNwN5p5EIERGnLqqg%3D%3D&pageNo={page_no}&numOfRows=100"

    print(url + "\n")

    response = requests.get(url).text.encode("utf-8")
    xml_content = bs4.BeautifulSoup(response, "lxml-xml")

    row_num = len(xml_content.find_all("item"))  # 수집된 content 수

    # 저장
    as2_err_cnt = 0
    for row in range(0, row_num):
        content[row + content_count] = {}
        content[row + content_count]["mega_nm"] = xml_content.find_all("as1")[row].text

        # 세종특별자치시 인 경우 시군구명이 없음
        if xml_content.find_all("as1")[row].text.__eq__('세종특별자치시'):
            content[row + content_count]["cty_nm"] = ""
            as2_err_cnt += 1
        else :
            content[row + content_count]["cty_nm"] = xml_content.find_all("as2")[row - as2_err_cnt].text

        content[row + content_count]["admi_nm"] = xml_content.find_all("as3")[row].text
        content[row + content_count]["zone_cd"] = xml_content.find_all("bjdCode")[row].text
        content[row + content_count]["apt_cd"] = xml_content.find_all("kaptCode")[row].text
        content[row + content_count]["apt_nm"] = xml_content.find_all("kaptName")[row].text

    content_count = len(content)
    total_cnt = int(xml_content.find("totalCount").text)  # 전체 개수
    total_page = total_cnt // 100 + 1  # 페이지 수

    # if total_page == page_no :
    if page_no == 10:
        break
    else:
        page_no += 1

    time.sleep(5)

df_content = pd.DataFrame(content).T
df_content.to_csv("data/apt_info/apt_info_list.csv", index=False, encoding="UTF-8")
