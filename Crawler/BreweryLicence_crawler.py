import pandas as pd
import re
import os
import time
import urllib.request
from operator import eq
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains

browser = webdriver.Chrome("C:\workspace\Python3\chromedriver_win32\chromedriver.exe")
browser.maximize_window()
url = "http://i.nts.go.kr/Default.asp"

browser.get(url)
time.sleep(10)

data = {}
cnt = 0
for i in range(1, len(browser.find_element_by_class_name("hbg_t").find_element_by_tag_name("ul").find_elements_by_tag_name("li")[10].find_element_by_tag_name("ul").find_elements_by_tag_name("li"))):
    browser.find_element_by_class_name("hbg_t").find_element_by_tag_name("ul").find_elements_by_tag_name("li")[10].find_element_by_tag_name("ul").find_elements_by_tag_name("li")[i].click()
    category = browser.find_element_by_class_name("hbg_t").find_element_by_tag_name("ul").find_elements_by_tag_name("li")[10].find_element_by_tag_name("ul").find_elements_by_tag_name("li")[i].text
    time.sleep(10)

    pageLen = int(re.sub("페이지수 : ", "", browser.find_element_by_class_name("listnum").text.split(",")[1]).split("/")[1])
    for k in range(0, pageLen):
        contentLen = len(browser.find_element_by_class_name("table_list").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr"))
        for j in range(0, contentLen):
            data[cnt] = {}
            browser.find_element_by_class_name("table_list").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[j].find_element_by_class_name("list_subject").find_element_by_tag_name("a").click()
            time.sleep(10)

            if eq(category, "탁주") or eq(category, "약주") or eq(category, "청주"):
                data[cnt]["ProductName"] = browser.find_element_by_class_name("board_view").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[0].find_element_by_class_name("view_subject").text
                data[cnt]["BreweryName"] = re.sub("제조장명 : ", "", browser.find_element_by_class_name("view_content").text.split("\n")[1])
                cnt += 1
                browser.back()
                time.sleep(10)

            elif eq(category, "맥주"):
                continue

            else:
                data[cnt]["ProductName"] = browser.find_element_by_class_name("board_view").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[0].find_element_by_class_name("view_subject").text.split("/")[1]
                data[cnt]["BreweryName"] = browser.find_element_by_class_name("board_view").find_element_by_tag_name("tbody").find_elements_by_tag_name("tr")[0].find_element_by_class_name("view_subject").text.split("/")[0]
                cnt += 1
                browser.back()
                time.sleep(10)

        # if (k+1)%10 == 0:
        #     if (k+1) == 10:
        #         browser.find_element_by_class_name("paging").find_elements_by_tag_name("a")[9].click()
        #     else:
        #         browser.find_element_by_class_name("paging").find_elements_by_tag_name("a")[10].click()
        # else:
        #     browser.find_element_by_class_name("paging").find_elements_by_tag_name("a")[k%10].click()

        if (k+1) < 10:
            browser.find_element_by_class_name("paging").find_elements_by_tag_name("a")[k % 10].click()
        elif (k+1) == 10:
            browser.find_element_by_class_name("paging").find_elements_by_tag_name("a")[9].click()
        else:
            if (k+1) % 10 == 0:
                browser.find_element_by_class_name("paging").find_elements_by_tag_name("a")[10].click()
            else:
                browser.find_element_by_class_name("paging").find_elements_by_tag_name("a")[k % 10].click()

        time.sleep(10)

dataDF = pd.DataFrame(data).T
dataDF.to_csv("BreweryLicence.csv", encoding="utf8", index=False)
browser.close()
