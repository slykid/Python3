import pandas as pd
import re
import os
import time
from operator import eq
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import urllib.request

browser = webdriver.Chrome("D:\workspace\Python\Python3\chromedriver.exe")
#browser = webdriver.Chrome("C:\workspace\Python3\chromedriver_win32\chromedriver.exe")

url = "http://thesool.com"

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

if os.path.exists("TheSoolBreweryDict") is False:
    os.mkdir("TheSoolBreweryDict")

browser.get(url)
time.sleep(10)

# 우리 술 찾기
action = ActionChains(browser)
action.move_to_element(browser.find_element_by_class_name("fusion-secondary-main-menu").find_element_by_class_name("gnb_wrap").find_element_by_class_name("gnb").find_element_by_class_name("dep1")).perform()
time.sleep(5)
listCnt = len(browser.find_element_by_class_name("gnb_wrap").find_element_by_class_name("gnb").find_element_by_class_name("dep1").find_elements_by_class_name("dep2")[3].find_elements_by_class_name("dep3")[1].find_elements_by_tag_name("li"))
data ={}
cnt = 0
errorCnt = 0

for i in range(0, listCnt):
    action = ActionChains(browser)
    action.move_to_element(browser.find_element_by_class_name("fusion-secondary-main-menu").find_element_by_class_name("fusion-row").find_element_by_class_name("gnb_wrap").find_element_by_class_name("gnb").find_element_by_id("gnbMenu")).perform()
    time.sleep(5)
    browser.find_element_by_class_name("fusion-secondary-main-menu").find_element_by_class_name("gnb_wrap").find_element_by_class_name("gnb").find_element_by_class_name("dep1").find_elements_by_class_name("dep2")[3].find_elements_by_class_name("dep3")[1].find_elements_by_tag_name("li")[i].find_element_by_tag_name("a").click()
    time.sleep(10)

    # 페이지 수 카운트
    pageCnt = int(browser.find_element_by_class_name("wrap_pagination").find_elements_by_tag_name("li")[len(browser.find_element_by_class_name("wrap_pagination").find_elements_by_tag_name("li"))-2].text)
    for j in range(0, pageCnt):
        # 목록 수 카운트
        contentCnt = len(browser.find_element_by_class_name("list_box").find_element_by_tag_name("ul").find_elements_by_tag_name("li"))
        for k in range(0, contentCnt):
            browser.find_element_by_class_name("list_box").find_element_by_tag_name("ul").find_elements_by_tag_name("li")[k].find_element_by_class_name("thum").find_element_by_tag_name("a").click()
            time.sleep(10)

            data[cnt-errorCnt] = {}
            try:
                data[cnt-errorCnt]["ProductName"] = re.sub("/", "", browser.find_element_by_class_name("cont").find_element_by_class_name("pc").find_element_by_tag_name("h4").find_element_by_tag_name("a").text)
            except:
                browser.back()
                errorCnt += 1
                continue

            browser.find_element_by_class_name("thum").find_element_by_class_name("img").find_element_by_tag_name("a").click()
            time.sleep(10)
            #browser.switch_to_window(browser.window_handles[1])
            link = browser.find_element_by_tag_name("img").get_property("src")
            isSuccess = download_photo(link, str("TheSoolBreweryDict/" + data[cnt-errorCnt]["ProductName"] + ".jpg"))
            if isSuccess == 0:
                data[cnt-errorCnt]["PictureURI"] = str("TheSoolBreweryDict/" + data[cnt-errorCnt]["ProductName"] + ".jpg")
            browser.back()
            time.sleep(10)

            for l in range(0, len(browser.find_element_by_class_name("cont").find_element_by_class_name("text_wrap").find_elements_by_class_name("text_box"))):
                if eq(browser.find_element_by_class_name("cont").find_element_by_class_name("text_wrap").find_elements_by_class_name("text_box")[l].find_element_by_tag_name("span").find_element_by_tag_name("strong").text , "종류"):
                    data[cnt-errorCnt]["Category"] = re.sub("종류 : ","",browser.find_element_by_class_name("cont").find_element_by_class_name("text_wrap").find_elements_by_class_name("text_box")[l].find_element_by_tag_name("span").text)

                elif eq(browser.find_element_by_class_name("cont").find_element_by_class_name("text_wrap").find_elements_by_class_name("text_box")[l].find_element_by_tag_name("span").find_element_by_tag_name("strong").text , "원재료"):
                    data[cnt-errorCnt]["Material"] = re.sub("원재료 : ","",browser.find_element_by_class_name("cont").find_element_by_class_name("text_wrap").find_elements_by_class_name("text_box")[l].find_element_by_tag_name("span").text)

                elif eq(browser.find_element_by_class_name("cont").find_element_by_class_name("text_wrap").find_elements_by_class_name("text_box")[l].find_element_by_tag_name("span").find_element_by_tag_name("strong").text , "알콜도수"):
                    data[cnt-errorCnt]["Alcohol"] = re.sub("알콜도수 : ","",browser.find_element_by_class_name("cont").find_element_by_class_name("text_wrap").find_elements_by_class_name("text_box")[l].find_element_by_tag_name("span").text)

                elif eq(browser.find_element_by_class_name("cont").find_element_by_class_name("text_wrap").find_elements_by_class_name("text_box")[l].find_element_by_tag_name("span").find_element_by_tag_name("strong").text , "용량"):
                    data[cnt-errorCnt]["Volume"] = re.sub("용량 : ","",browser.find_element_by_class_name("cont").find_element_by_class_name("text_wrap").find_elements_by_class_name("text_box")[l].find_element_by_tag_name("span").text)

            try:
                data[cnt-errorCnt]["Overview"] = browser.find_elements_by_class_name("content_text")[0].find_element_by_class_name("main_text").text
            except:
                data[cnt-errorCnt]["Overview"] = ""

            try:
                data[cnt-errorCnt]["Detail"] = browser.find_elements_by_class_name("content_text")[1].find_element_by_class_name("main_text").text
            except:
                data[cnt-errorCnt]["Detail"] = ""

            cnt += 1
            browser.back()
            time.sleep(10)

        if j < pageCnt:
            browser.find_element_by_class_name("wrap_pagination").find_elements_by_tag_name("li")[len(browser.find_element_by_class_name("wrap_pagination").find_elements_by_tag_name("li")) - 1].click()
            time.sleep(10)

dataDF = pd.DataFrame(data).T
dataDF = dataDF[["ProductName", "Category", "Material", "Alcohol", "Volume", "Overview", "Detail", "PictureURI"]]
dataDF.to_csv("TheSool_brewery.csv", encoding="UTF-8", index=False)

browser.close()