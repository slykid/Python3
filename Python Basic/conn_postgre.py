import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns

# Postgres connection 설정
conn = psycopg2.connect(host='localhost', database='postgres', user='postgres', password='password', port=5432)

# cursor 객체 생성
cur = conn.cursor()

# SQL 문 실행
cur.execute("select * from test_db")

# 결과 가져오기
result = cur.fetchall()
print(result)
# [(Decimal('1'), '김길현'), (Decimal('2'), '유재석'), (Decimal('3'), '하동훈'), (Decimal('4'), '송지효')]

# 데이터프레임으로 변환
resultDF = pd.DataFrame(result)
print(resultDF)
#    0    1
# 0  1  김길현
# 1  2  유재석
# 2  3  하동훈
# 3  4  송지효

resultDF.columns = [desc[0]for desc in cur.description]  # 컬럼 정보 -> 데이터 프레임에 추가
print(resultDF)
#   no name
# 0  1  김길현
# 1  2  유재석
# 2  3  하동훈
# 3  4  송지효

resultDF["no"] = resultDF["no"].astype("int32")  # 숫자 정보 형변환 (Decimal -> int32)

# 막대그래프 시각화
sns.barplot(resultDF["no"])
