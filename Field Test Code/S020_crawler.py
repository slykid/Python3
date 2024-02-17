import pandas as pd
import re
import time
from operator import eq
from selenium import webdriver

url = "https://www.naver.com/"
input = pd.read_csv("data/scrapInput.csv", encoding="euc=kr")

searchList = []
for i in range(0, len(input["STORE_NM"])):
    addr = ''
    if re.search("-", input["ADDR"][i]) is not None:
        if eq(input["ADDR"][i].split(",")[0].split("-")[1], '0'):
           addr = input["ADDR"][i].split(",")[0].split("-")[0]
        else :
            addr = input["ADDR"][i].split(",")[0]
    else :
        addr = input["ADDR"][i]

    searchList.append(input["STORE_NM"][i] + " " + addr)

browser = webdriver.Chrome("driver/chromedriver")
browser.maximize_window()

data = {}
contentCnt = 0

start = time.time()  # 시간 측정
for query in searchList:
    batch_start = time.time()
    browser.get(url)
    time.sleep(10)

    # 검색어 입력
    browser.find_element_by_class_name("green_window").find_element_by_tag_name("input").send_keys(query)
    browser.find_element_by_class_name("green_window").submit()
    time.sleep(10)

    # 지도 태그 클릭 및 웹 페이지 전환
    data[contentCnt] = {}
    data[contentCnt]["store_nm"] = input["STORE_NM"][contentCnt]
    data[contentCnt]["upjong_desc"] = input["UPJONG_DESC"][contentCnt]

    try:
        browser.find_element_by_class_name("greenmap").find_element_by_class_name("local_map").find_element_by_class_name("detail").find_element_by_class_name("correct_wrap").find_element_by_class_name("tit_area").find_element_by_tag_name("a").click()
        time.sleep(10)
        browser.switch_to.window(browser.window_handles[1])
        time.sleep(10)

        try:
            data[contentCnt]["category"] = browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[0].find_element_by_class_name("biz_name_area").find_element_by_class_name("category").text
            listCnt = len(browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item"))

            for i in range(0, listCnt):
                if eq("전화", browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[i].find_element_by_class_name("tit").get_attribute("aria-label")):
                    data[contentCnt]["tel"] = browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[i].find_element_by_class_name("txt").text
                if eq("주소", browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[i].find_element_by_class_name("tit").get_attribute("aria-label")):
                    data[contentCnt]["addr"] = browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[i].find_element_by_class_name("txt").text.split("\n")[0]
                if eq("메뉴", browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[i].find_element_by_class_name("tit").get_attribute("aria-label")):
                    data[contentCnt]["menu"] = browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[i].find_element_by_class_name("txt").text.split("\n")[1]
                    data[contentCnt]["price"] = browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[i].find_element_by_class_name("txt").text.split("\n")[0]

        except:
            jibunCnt = 0
            for num in range(0, len(browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_class_name("spm"))):
                if eq("주소", browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_class_name("spm")[num].text):
                    if re.search("지번", browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dd")[num].text.split("\n")[1]) is not None:
                        data[contentCnt]["addr"] = browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dd")[num].text.split("\n")[0]
                        jibunCnt += 1
                    else:
                        data[contentCnt]["addr"] = browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dd")[num-jibunCnt].text

                elif eq("전화번호", browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_class_name("spm")[num].text):
                    data[contentCnt]["tel"] = browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dd")[num-jibunCnt].text

                elif eq("분류", browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_class_name("spm")[num].text):
                    data[contentCnt]["category"] = browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dd")[num-jibunCnt].text

                elif eq("메뉴", browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_class_name("spm")[num].text):
                    data[contentCnt]["menu"] = browser.find_element_by_class_name("local_info").find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dd")[num-jibunCnt].text

                data[contentCnt]["price"] = ""

        time.sleep(10)
        browser.close()
        time.sleep(5)
        browser.switch_to.window(browser.window_handles[0])
        time.sleep(5)

    except:
        data[contentCnt]["category"] = ""
        data[contentCnt]["tel"] = ""
        data[contentCnt]["addr"] = ""
        data[contentCnt]["menu"] = ""
        data[contentCnt]["price"] = ""

    try:
        if browser.window_handles[1] is not None:
            browser.switch_to.window(browser.window_handles[1])
            browser.close()
    except:
        time.sleep(1)
    finally:
        browser.switch_to.window(browser.window_handles[0])
        time.sleep(5)
    print(str(contentCnt+1) + "번 : " + input["STORE_NM"][contentCnt] + " 의 수집이 완료되었습니다.")
    print("Batch Runtime : ", str(round(time.time() - batch_start), 2), " sec | 작업 진행률 : ", str( round(contentCnt * 100 / len(input["STORE_NM"]), 2) ) + "%\n")
    contentCnt += 1

print("Crawler Runtime : ", str(round(time.time() - start), 2), "sec")

dataDF = pd.DataFrame(data).T
dataDF = dataDF[["store_nm", "category", "upjong_desc", "addr", "tel", "menu", "price"]]
dataDF.to_csv("S020_crawler_result.csv", index=False, encoding="utf8")

browser.close()
