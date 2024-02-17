import pandas as pd
import re
import os
import time
from operator import eq
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains

#browser = webdriver.Chrome("D:\workspace\Python\Python3\chromedriver.exe")
browser = webdriver.Chrome("C:\workspace\Python3\chromedriver_win32\chromedriver.exe")

url = "https://www.google.com/maps/d/viewer?mid=1XZ0LD5WJn9ZMi_8KXnBBCDcmCCA&hl=en_US&ll=36.03736199438497%2C127.7078436476562&z=6"

browser.get(url)

for i in range(0, len(browser.find_elements_by_class_name("HzV7m-pbTTYe-KoToPc-ornU0b"))):
    browser.find_elements_by_class_name("HzV7m-pbTTYe-KoToPc-ornU0b")[i].click()

data = {}
listLen = len(browser.find_elements_by_class_name("HzV7m-pbTTYe-ibnC6b-V67aGc"))
for i in range(1, listLen):
    browser.find_elements_by_class_name("HzV7m-pbTTYe-ibnC6b-V67aGc")[i].click()
    time.sleep(10)

    data[i-1] = {}
    data[i-1]["BreweryName"] = browser.find_element_by_css_selector("#featurecardPanel > div > div > div.qqvbed-bN97Pc > div.qqvbed-UmHwN").find_elements_by_class_name("qqvbed-p83tee")[0].find_element_by_class_name("qqvbed-p83tee-lTBxed").text

    try:
        data[i-1]["Description"] = browser.find_element_by_css_selector("#featurecardPanel > div > div > div.qqvbed-bN97Pc > div.qqvbed-UmHwN").find_elements_by_class_name("qqvbed-p83tee")[1].find_element_by_class_name("qqvbed-p83tee-lTBxed").text
    except:
        data[i-1]["Description"] = ""

    try:
        data[i-1]["Address"] = browser.find_element_by_css_selector("#featurecardPanel > div > div > div.qqvbed-bN97Pc > div.qqvbed-VTkLkc.fO2voc-jRmmHf-LJTIlf").find_element_by_class_name("fO2voc-jRmmHf-MZArnb-Q7Zjwb").text
    except:
        data[i-1]["Address"] = ""

    browser.find_element_by_css_selector("#featurecardPanel > div > div > div.qqvbed-tJHJj > div.HzV7m-tJHJj-LgbsSe-haAclf.qqvbed-a4fUwd-LgbsSe-haAclf > div").click()
    time.sleep(10)

dataDF = pd.DataFrame(data).T
dataDF = dataDF[["BreweryName", "Description", "Address"]]
dataDF.to_csv("magmap_brewery_info.csv", encoding="UTF-8", index=False)