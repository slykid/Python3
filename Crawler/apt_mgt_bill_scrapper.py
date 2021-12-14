import requests, bs4
import pandas as pd
import time

# 변수 설정
service_key = "n1tWqYKnI0IwXCLBdC8U3oGZc1w5Tb9KsDh2vPyZ44Sl5OyMmv%2FDMCk%2BeUgWWSGFWWxFbQNwN5p5EIERGnLqqg%3D%3D"
contents = {}
cnt = 0

# 아파트 단지목록 로드
apt_list = pd.read_csv("data/apt_info/apt_info_list.csv", encoding="UTF-8")

for _apt_cd in apt_list["apt_cd"]:
    response = requests.get("http://apis.data.go.kr/1613000/AptBasisInfoService1/getAphusBassInfo" + "?serviceKey=" + service_key + "&kaptCode=" + _apt_cd).text.encode("utf-8")
    xml_content = bs4.BeautifulSoup(response, "lxml-xml")

    print(_apt_cd)

    contents[cnt] = {}

    ## 단지코드
    try: contents[cnt]["apt_code"] = xml_content.find("kaptCode").text
    except: contents[cnt]["apt_code"] = _apt_cd

    ## 단지명
    try: contents[cnt]["apt_nm"] = xml_content.find("kaptName").text
    except: contents[cnt]["apt_nm"] = ""

    ## 단지분류
    try: contents[cnt]["apt_kind"] = xml_content.find("codeAptNm").text
    except: contents[cnt]["apt_kind"] = ""

    ## 법정동 코드
    try: contents[cnt]["zone_cd"] = xml_content.find("bjdCode").text
    except : contents[cnt]["zone_cd"] = ""

    ## 법정동 주소
    try: contents[cnt]["zone_addr"] = xml_content.find("kaptAddr").text
    except: contents[cnt]["zone_addr"] = ""

    ## 도로명 주소
    try: contents[cnt]["new_addr"] = xml_content.find("doroJuso").text
    except: contents[cnt]["new_addr"] = ""

    ## 건축물대장 연면적
    try: contents[cnt]["bld_ldg_area"] = float(xml_content.find("kaptTarea").text)
    except: contents[cnt]["bld_ldg_area"] = ""

    ## 단지 전용면적합
    try: contents[cnt]["area_private_area"] = float(xml_content.find("privArea").text)
    except: contents[cnt]["area_private_area"] = ""

    ## 동수
    try: contents[cnt]["dong_cnt"] = int(xml_content.find("kaptDongCnt").text)
    except: contents[cnt]["dong_cnt"] = ""

    ## 세대수
    try: contents[cnt]["hous_cnt"] = int(xml_content.find("kaptdaCnt").text)
    except: contents[cnt]["hous_cnt"] = ""

    ## 호 수
    try: contents[cnt]["ho_cnt"] = int(xml_content.find("hoCnt").text)
    except: contents[cnt]["ho_cnt"] = ""

    ## 시공사
    try: contents[cnt]["build_company"] = xml_content.find("kaptBcompany").text
    except: contents[cnt]["build_company"] = ""

    ## 시행사
    try: contents[cnt]["apply_company"] = xml_content.find("kaptAcompany").text
    except: contents[cnt]["apply_company"] = ""

    ## 분양형태
    try: contents[cnt]["sale_kind"] = xml_content.find("codeSaleNm").text
    except: contents[cnt]["sale_kind"] = ""

    ## 난방방식
    try: contents[cnt]["heat_kind"] = xml_content.find("codeHeatNm").text
    except: contents[cnt]["heat_kind"] = ""

    ## 복도유형
    try: contents[cnt]["hall_kind"] = xml_content.find("codeHallNm").text
    except: contents[cnt]["hall_kind"] = ""

    ## 관리사무소 연락처
    try: contents[cnt]["bld_mgt_tel"] = xml_content.find("kaptTel").text
    except: contents[cnt]["bld_mgt_tel"] = ""

    ## 관리사무소 팩스
    try: contents[cnt]["bld_mgt_fax"] = xml_content.find("kaptFax").text
    except: contents[cnt]["bld_mgt_fax"] = ""

    ## 홈페이지 주소
    try: contents[cnt]["url"] = xml_content.find("kaptUrl").text
    except: contents[cnt]["url"] = ""

    ## 사용승인일
    try: contents[cnt]["use_date"] = xml_content.find("kaptUsedate").text
    except: contents[cnt]["use_date"] = ""

    ## 관리비 부과면적
    try: contents[cnt]["mgt_bill_area"] = float(xml_content.find("kaptMarea").text)
    except: contents[cnt]["mgt_bill_area"] = ""

    ## 전용면적별 세대현황 (60m2 이하)
    try: contents[cnt]["private_cnt_under60"] = int(xml_content.find("kaptMparea_60").text)
    except: contents[cnt]["private_cnt_under60"] = ""

    ## 전용면적별 세대현황 (60 ~ 85m2 이하)
    try: contents[cnt]["private_cnt_under85"] = int(xml_content.find("kaptMparea_85").text)
    except: contents[cnt]["private_cnt_under85"] = ""

    ## 전용면적별 세대현황 (85 ~ 135m2 이하)
    try: contents[cnt]["private_cnt_under135"] = int(xml_content.find("kaptMparea_135").text)
    except: contents[cnt]["private_cnt_under135"] = ""

    ## 전용면적별 세대현황 (135m2 초과)
    try: contents[cnt]["private_cnt_over135"] = int(xml_content.find("kaptMparea_136").text)
    except: contents[cnt]["private_cnt_over135"] = ""

    cnt += 1

    time.sleep(5)

# 데이터프레임 생성
df_content = pd.DataFrame(contents).T
df_content.to_csv("data/apt_info/apt_mgt_info.csv", index=False, encoding="UTF-8")