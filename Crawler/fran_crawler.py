import pandas as pd
import re
import os
import time
from operator import eq
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome("driver/chromedriver.exe")
driver.maximize_window()
driver.get("https://franchise.ftc.go.kr/mnu/00013/program/userRqst/list.do")



