import os
import numpy as np
import pandas as pd
import re

# Q1. 2020년 데이터만 로드하시오.
# 경기, 선수, 팀 데이터는 파일인코딩을 'CP949' 로 설정 후 로드한다.
## 개인타자(df_hitter), 개인투수 (df_pitcher),  경기(df_game), 등록선수(df_entry), 선수(df_player),
## 팀(df_team), 팀타자(df_team_hitter), 팀투수(df_team_pitcher)
df_hitter =
df_pitcher =
df_game =
df_entry =
df_player =
df_team =
df_team_hitter =
df_team_pitcher =

# Q2. 개인투수 데이터의 컬럼별 자료형을 출력하시오


# Q3. 개인타자 중 SK 소속인 타자들의 ID(P_ID)를 df_hitter_sk 에 저장하시오.
#     (컬럼명은 P_ID 로 지정한다.)
df_hitter_sk =

# Q4. 개인타자 중 타율(HIT)이 3.0 이상인 타자들만 df_hitter_3_0 에 저장하고 인덱스를 재설정하시오.
#     재설정 시에는 df.reset_index() 를 사용하며, 사용 후에는 index 라는 컬럼이 생성됨
#     생성된 index 컬럼은 drop 시킴
df_hitter_3_0 =

# Q5. 선수 데이터 중 외국인 선수들의 경우 한화가 아닌 달러로 표시되어있다.
#     동일하게 하기 위해 모든 선수의 연봉을 한화로 변경시키시오.
#     컬럼명은 "MONEY_KOR" 로 하며, 1달러 = 1200원 으로 한다.
#     "만원", "달러" 등의 단어는 re 모듈의 search() 함수로 검색 후, sub() 함수를 이용해서 제거한다.





