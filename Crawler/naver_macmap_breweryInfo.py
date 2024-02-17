import pandas as pd
import re
import os
import time
from operator import eq
from selenium import webdriver
import urllib.request

listDF = pd.read_csv("magmap_brewery_info.csv", encoding="UTF8")
queries = list(listDF["BreweryName"])

#browser = webdriver.Chrome("D:\workspace\Python\Python3\chromedriver.exe")
browser = webdriver.Chrome("C:\workspace\Python3\chromedriver_win32\chromedriver.exe")

url = "http://www.naver.com"

# 이미지 다운로드를 위한 함수
def download_photo(img_url, filename):
    downloaded_image = open(filename, "wb")
    try:
        image_on_web = urllib.request.urlopen(img_url)

        while True:
            buf = image_on_web.read(100000000)
            if len(buf)==0:
                break
            downloaded_image.write(buf)

        downloaded_image.close()
        image_on_web.close()

        return 0

    except:
        downloaded_image.close()
        print("Exception: Access Disabled")
        os.remove(filename)
        return -1

if os.path.exists("MacMapBreweryDict") is False:
    os.mkdir("MacMapBreweryDict")

data = {}
errorCnt = 0
for i in range(0, len(queries)):
    browser.get(url)
    time.sleep(10)

    data[i-errorCnt] = {}

    browser.find_element_by_class_name("green_window").find_element_by_id("query").send_keys(queries[i])
    browser.find_element_by_class_name("green_window").submit()
    time.sleep(10)

    try:
        browser.find_element_by_class_name("greenmap").find_element_by_class_name("local_map").find_element_by_class_name("detail").find_element_by_class_name("correct_wrap").find_element_by_class_name("tit_area").find_element_by_tag_name("a").click()
        time.sleep(10)
        browser.switch_to.window(browser.window_handles[1])

    except:
        errorCnt += 1
        continue

    if eq(queries[i], "쓰리몽키즈 양조장") or eq(queries[i], "제스피 제조장") or eq(queries[i], "제주지앵") or eq(queries[i], "더 세를라잇 브루잉") or eq(queries[i], "굿맨브루어리") or \
            eq(queries[i], "더 핸드 앤 몰트 (양조장)") or eq(queries[i], "앰비션 브루어리") or eq(queries[i], "뱅크크릭브루잉") or eq(queries[i], "블루웨일브루하우스") or \
            eq(queries[i], "화이트 크로우 브루잉") or eq(queries[i], "안동브루잉컴퍼니") or eq(queries[i], "켈슈브로이") or eq(queries[i], "크래프트 인조이 브루잉"):
        data[i-errorCnt]["BreweryName"] = browser.find_element_by_class_name("title_area").find_element_by_class_name("title").text

        listLen = len(browser.find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dt"))
        for j in range(0, listLen):
            if eq("주소",browser.find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dt")[j].text):
                data[i-errorCnt]["Address"] = browser.find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dd")[j].text

            elif eq("전화번호",browser.find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dt")[j].text):
                data[i-errorCnt]["Phone"] = browser.find_element_by_class_name("local_info_detail").find_elements_by_tag_name("dd")[j].text

        try:
            data[i-errorCnt]["Facility"] = re.sub("\n", " ",browser.find_element_by_id("_baseTabContents").find_element_by_class_name("section_detail_info").text)
        except:
            data[i - errorCnt]["Facility"] = ""

        try:
            data[i-errorCnt]["Overview"] = browser.find_element_by_css_selector("#_baseInfo > div:nth-child(2)").find_element_by_tag_name("p").text
        except:
            data[i - errorCnt]["Overview"] = ""

        try:
            browser.find_element_by_class_name("local_info_thumb").find_element_by_tag_name("a").click()
            link = browser.find_element_by_class_name("local_info_thumb").find_element_by_tag_name("a").find_element_by_tag_name("img").get_property("src")
            isSuccess = download_photo(link, str("MacMapBreweryDict/" + data[i-errorCnt]["BreweryName"] + ".jpg"))
            if isSuccess == 0:
                data[i-errorCnt]["PictureURI"] = str("MacMapBreweryDict/" + data[i-errorCnt]["BreweryName"] + ".jpg")
        except:
            data[i - errorCnt]["PictureURI"] = ""
            time.sleep(3)

        browser.close()
        browser.switch_to.window(browser.window_handles[0])
        time.sleep(10)

    else:
        data[i-errorCnt]["BreweryName"] = browser.find_element_by_class_name("ct_box_area").find_element_by_class_name("biz_name_area").find_element_by_tag_name("strong").text

        listLen = len(browser.find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item"))
        for j in range(0, listLen):
            if eq("전화", browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[j].find_element_by_tag_name("span").get_attribute("aria-label")):
                data[i-errorCnt]["Phone"] = browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[j].find_element_by_class_name("txt").text

            elif eq("주소", browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[j].find_element_by_tag_name("span").get_attribute("aria-label")):
                data[i-errorCnt]["Address"] = browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[j].find_element_by_class_name("txt").find_element_by_class_name("list_address").find_elements_by_tag_name("li")[0].text

            elif eq("편의시설", browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[j].find_element_by_tag_name("span").get_attribute("aria-label")):
                data[i-errorCnt]["Facility"] = browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[j].find_element_by_class_name("txt").text

            elif eq("업체설명", browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[j].find_element_by_tag_name("span").get_attribute("aria-label")):
                data[i-errorCnt]["Overview"] = browser.find_element_by_class_name("content").find_elements_by_class_name("ct_box_area")[1].find_element_by_class_name("bizinfo_area").find_element_by_class_name("list_bizinfo").find_elements_by_class_name("list_item")[j].find_element_by_class_name("txt").text

        try:
            browser.find_element_by_class_name("flick_container").find_element_by_class_name("btn_view_all").click()
            link = browser.find_element_by_class_name("flick_content").find_element_by_class_name("thumb_area").find_element_by_class_name("thumb").find_element_by_tag_name("img").get_property("src")
            isSuccess = download_photo(link, str("MacMapBreweryDict/" + data[i-errorCnt]["BreweryName"] + ".jpg"))
            if isSuccess == 0:
                data[i-errorCnt]["PictureURI"] = str("MacMapBreweryDict/" + data[i-errorCnt]["BreweryName"] + ".jpg")
        except:
            data[i - errorCnt]["PictureURI"] = ""
        browser.close()
        browser.switch_to.window(browser.window_handles[0])
        time.sleep(10)

dataDF = pd.DataFrame(data).T
dataDF = dataDF[["BreweryName", "Phone", "Address", "Facility", "Overview"]]
dataDF.to_csv("naver_macmap_breweryInfo.csv", encoding="UTF8", index=False)
browser.close()