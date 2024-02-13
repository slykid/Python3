import pandas as pd
import re
import os
import uuid
import time
from operator import eq
import urllib.request as req
from selenium import webdriver


#browser = webdriver.Chrome("D:\workspace\Python3\\venv\Lib\chromedriver_win32\chromedriver.exe")
browser = webdriver.Chrome("C:\workspace\Python3\chromedriver_win32\chromedriver.exe")

url = "http://www.naver.com"

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

browser.find_element_by_class_name("themecast_list").find_element_by_class_name("tl_title").find_element_by_class_name("tt_links").find_elements_by_tag_name("li")[4].find_element_by_class_name("tt_la").click()
time.sleep(10)

# 술 종류 리스트
brewer={}
for i in range(1, len(browser.find_element_by_class_name("category_horizontal_circle").find_element_by_id("categoryScrollWrapper").find_elements_by_tag_name("li"))):
    brewerCategory = browser.find_element_by_class_name("category_horizontal_circle").find_element_by_id("categoryScrollWrapper").find_elements_by_tag_name("li")[i].find_element_by_class_name("text").text

    browser.find_element_by_class_name("category_horizontal_circle").find_element_by_id("categoryScrollWrapper").find_elements_by_tag_name("li")[i].find_element_by_tag_name("a").click()

    prevHeight = browser.execute_script("return document.body.scrollHeight")
    while True:
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        newHeight = browser.execute_script("return document.body.scrollHeight")
        if prevHeight == newHeight:
            break
        else:
            prevHeight = newHeight
            time.sleep(2)

    del prevHeight
    del newHeight

    browser.execute_script("window.scrollTo(0, 0);")

    prevListCount = 0
    listLen = len(browser.find_element_by_class_name("common_list_static").find_elements_by_class_name("list"))

    storeList =[]
    for j in range(0, listLen):
        try:
            storeName = browser.find_element_by_css_selector("#grid > li:nth-child(" + str(j + 1) + ")").find_element_by_class_name("store_link").find_element_by_class_name("title").text
        except:
            continue

        if storeName in storeList:
            continue
        else:
            storeList.append(storeName)
            browser.find_element_by_css_selector("#grid > li:nth-child(" + str(j+1) + ")").find_element_by_class_name("store_link").click()
            time.sleep(10)

            productLen = len(browser.find_element_by_class_name("product_set").find_elements_by_tag_name("li"))
            for k in range(0, productLen):
                brewer[prevListCount+k] = {}

                brewer[prevListCount + k]["StoreName"] = storeName

                try:
                    brewer[prevListCount+k]["Info"] = browser.find_element_by_class_name("_1g1GQ99wGd").text
                except:
                    brewer[prevListCount + k]["Info"] = ""
                    time.sleep(1)
                try:
                    brewer[prevListCount+k]["Phone"] = browser.find_element_by_class_name("_3PtDfR2bEb").find_element_by_tag_name("li").text.split("\n")[0]
                except:
                    brewer[prevListCount + k]["Phone"] = ""
                    time.sleep(1)

                try:
                    brewer[prevListCount+k]["ProductName"] = browser.find_element_by_class_name("product_set").find_elements_by_tag_name("li")[k].find_element_by_class_name("_3qk21YA7xo").find_element_by_tag_name("p").text
                except:
                    brewer[prevListCount + k]["ProductName"] = ""
                    time.sleep(1)

                try:
                    brewer[prevListCount+k]["Price"] = int(re.sub(",", "",re.sub("원", "",browser.find_element_by_class_name("product_set").find_elements_by_tag_name("li")[k].find_element_by_class_name("_3qk21YA7xo").find_element_by_class_name("price").text)))
                except:
                    brewer[prevListCount + k]["Price"] = ""
                    time.sleep(1)

                try:
                    brewer[prevListCount+k]["Review"] = browser.find_element_by_class_name("product_set").find_elements_by_tag_name("li")[k].find_element_by_class_name("_1uxCCJErwK").find_elements_by_tag_name("dd")[0].text
                except:
                    brewer[prevListCount + k]["Review"] = ""
                    time.sleep(1)

                try:
                    brewer[prevListCount+k]["Preference"] = float(browser.find_element_by_class_name("product_set").find_elements_by_tag_name("li")[k].find_element_by_class_name("_1uxCCJErwK").find_elements_by_tag_name("dd")[1].text.split("/")[0])
                except:
                    brewer[prevListCount + k]["Preference"] = ""
                    time.sleep(1)

            prevListCount += productLen
            browser.back()
            time.sleep(10)

brewerDF = pd.DataFrame(brewer).T
brewerDF = brewerDF[["StoreName", "Info", "Phone", "ProductName", "Price", "Review", "Preference"]]
brewerDF.to_csv("Data/brewerDF.csv", encoding="UTF-8", index=False)
