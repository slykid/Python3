from bs4 import BeautifulSoup
import datetime
import requests
import time
import random
from collections import defaultdict
import pandas as pd

api_col = ['rownum', 'opnsfteamcode', 'mgtno', 'opnsvcid', 'updategbn', 'updatedt', 'opnsvcnm', 'bplcnm',
'sitepostno', 'sitewhladdr', 'rdnpostno', 'rdnwhladdr', 'sitearea', 'apvpermymd', 'apvcancelymd', 'dcbymd',
'clgstdt', 'clgenddt', 'ropnymd', 'trdstategbn', 'trdstatenm', 'dtlstategbn', 'dtlstatenm', 'x', 'y',
'lastmodts',
'uptaenm', 'sitetel']

col_name_edit = ['num', 'gov_code', 'admin_no', 's_id', 'renew_gubun', 'renew_dt', 's_name', 'company',
'post_no', 'addr', 'road_postno', 'road_addr', 'area', 'license_dt', 'license_x_dt', 'close_dt',
'close_temp_startdt',
'close_temp_findt', 'reopen_dt', 'status', 'status_nm', 'status_detail', 'status_detail_nm', 'x_axis',
'y_axis',
'last_moddt', 'biz_type_nm', 'phone']

api_key = "34NwYHef1K8E4=p9yX8SsKyFlx0Rr8THyI035Y5EZaQ="
url = f"http://www.localdata.go.kr/platform/rest/TO0/openDataApi?authKey={api_key}"

page_index = 1

api_dict = defaultdict(list)

# 날짜계산
today = datetime.date.today()

before_yesterday = today - datetime.timedelta(days=2)
day = before_yesterday.strftime("%d/%m/%Y").split("/")[0]
month = before_yesterday.strftime("%d/%m/%Y").split("/")[1]
year = before_yesterday.strftime("%d/%m/%Y").split("/")[2]

date = year + month + day

while True:
    param = f"&lastModTsBgn={date}&lastModTsEnd={date}&pageSize=500&pageIndex={page_index}"
    res = requests.get(url + param)
    page_index += 1

    soup = BeautifulSoup(res.text, 'html.parser')

    for col in api_col:
        for i in soup.find_all(col)[1:]:
            api_dict[col].append(i.text)

# 마지막 페이지면 탈출
    if page_index == (int(soup.totalcount.text) // 500) + 1:
        break

res_df = pd.DataFrame(api_dict)

# 컬럼명 변경
res_df.columns = col_name_edit

# 요청날짜 컬럼 추가
res_df["request_date"] = date

# 저장
res_df.to_parquet(f"result/local_data/{date}.parquet")