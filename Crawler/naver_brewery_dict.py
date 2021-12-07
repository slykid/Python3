import pandas as pd
import re
import os
import time
from operator import eq
from selenium import webdriver
import urllib.request

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

browser.get(url)
time.sleep(10)

try:
    browser.find_element_by_id("PM_ID_themelist").find_elements_by_tag_name("li")[1].click()
except:
    while True:
        browser.find_element_by_class_name("ac_btn_prev").click()
        if eq(browser.find_elements_by_class_name("rolling-panel")[1].text,"푸드") is True:
            break
time.sleep(10)

browser.find_element_by_id("PM_ID_themelist").find_elements_by_tag_name("li")[1].click()
time.sleep(10)

# 전통주 피드 클릭
browser.find_element_by_class_name("themecast_list").find_elements_by_class_name("tl_default")[len(browser.find_element_by_class_name("themecast_list").find_elements_by_class_name("tl_default"))-1].find_element_by_tag_name("a").click()
time.sleep(10)

# 한국 전통주 백과 클릭
browser.find_element_by_class_name("headword_title").find_element_by_class_name("cite").find_element_by_tag_name("a").click()
time.sleep(10)

# 전체 페이지 수 확인
returnPageUrl = browser.current_url
beforeurl = ""
url = ""
while(True):
    url = browser.current_url
    if eq(url, beforeurl):
        break
    else:
        beforeurl = url
        browser.find_element_by_class_name("paginate").find_element_by_class_name("next").find_element_by_tag_name("a").click()
        time.sleep(10)

pageLen = int(browser.find_element_by_class_name("paginate").find_element_by_tag_name("strong").text)
browser.get(returnPageUrl)
time.sleep(10)

if os.path.exists("BreweryDict") is False:
    os.mkdir("BreweryDict")

# 각 리스트 내용 클릭 및 크롤링
content = {}
contentLen = 0
for i in range(0, pageLen):
    articleLen = len(browser.find_element_by_class_name("list_wrap").find_element_by_class_name("content_list").find_elements_by_class_name("subject"))

    for j in range(0, articleLen):
        browser.find_element_by_class_name("list_wrap").find_element_by_class_name("content_list").find_elements_by_class_name("subject")[j].find_element_by_class_name("title").find_element_by_tag_name("a").click()
        time.sleep(10)

        content[contentLen] = {}
        content[contentLen]["ProductName"] = browser.find_element_by_class_name("headword_title").find_element_by_class_name("headword").text

        try:
            browser.find_element_by_class_name("size_ct_v2").find_element_by_class_name("att_type").find_element_by_class_name("inner_att_type").find_element_by_class_name("thmb").find_element_by_class_name("img_box").find_element_by_tag_name("a").click()
            time.sleep(10)
            browser.switch_to_window(browser.window_handles[1])
            link = browser.find_element_by_class_name("big_img").find_element_by_class_name("img_box").find_element_by_class_name("img_area").find_element_by_tag_name("img").get_property("src")
            isSuccess = download_photo(link, str("BreweryDict/"+content[contentLen]["ProductName"]+".jpg"))
            if isSuccess == 0:
                content[contentLen]["PictureURI"] = str("BreweryDict/"+content[contentLen]["ProductName"]+".jpg")
            browser.close()
            browser.switch_to_window(browser.window_handles[0])
            time.sleep(5)
        except:
            time.sleep(5)

        try:
            content[contentLen]["Overview"] = browser.find_element_by_class_name("headword_title").find_element_by_class_name("desc").text
        except:
            content[contentLen]["Overview"] = ""

        try:
            content[contentLen]["Summary"] = browser.find_element_by_class_name("summary_area").text
        except:
            content[contentLen]["Summary"] = ""

        tableLen = len(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr"))
        for k in range(0, tableLen):
            if eq(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("th").text, "주종"):
                content[contentLen]["Category"] = browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("td").text

            if eq(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("th").text, "도수"):
                content[contentLen]["Alcohol"] = browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("td").text

            if eq(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("th").text, "용량"):
                content[contentLen]["Volume"] = browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("td").text

            if eq(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("th").text, "가격"):
                content[contentLen]["Price"] = browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("td").text

            if eq(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("th").text, "제조사"):
                content[contentLen]["Company"] = browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("td").text

            if eq(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("th").text, "대표자명"):
                content[contentLen]["CEO"] = browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("td").text

            if eq(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("th").text, "주소"):
                content[contentLen]["Address"] = browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("td").text

            if eq(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("th").text, "연락처"):
                content[contentLen]["Phone"] = browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("td").text

            if eq(browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("th").text, "온라인스토어"):
                content[contentLen]["OnlineShop"] = browser.find_element_by_class_name("tmp_profile_tb").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[k].find_element_by_tag_name("td").text

        del tableLen
        contentLen += 1
        browser.back()
        time.sleep(10)

    if i <= 7:
        browser.find_element_by_id("container").find_element_by_id("content").find_element_by_class_name("paginate").find_element_by_class_name("next").find_element_by_tag_name("a").click()
    elif i > 7 and i < 17:
        browser.find_element_by_id("container").find_element_by_id("content").find_element_by_class_name("paginate").find_elements_by_tag_name("a")[6].click()
    elif i >= 17:
        browser.find_element_by_id("container").find_element_by_id("content").find_element_by_class_name("paginate").find_elements_by_tag_name("a")[i+1-pageLen].click()
    time.sleep(10)

contentDF = pd.DataFrame(content).T
contentDF = contentDF[["ProductName", "Category", "Alcohol", "Volume", "Price", "Overview", "Summary", "CEO", "Address", "Phone", "OnlineShop", "PictureURI"]]
contentDF.to_csv("naver_brewery_dict.csv", encoding="UTF8", index=False)