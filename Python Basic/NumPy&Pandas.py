import numpy as np

# 1. Numpy
## 1) ndarray
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)

arr2.ndim  # ndarray 의 차원 확인
arr2.shape  # ndarray 의 형태
arr2.dtype  # ndarray 의 데이터타입을 추론해 출력함
print(arr2)

## 2) 표준배열 생성함수
data3 = np.ones(3)  # 1 x 3 의 배열로 모든 요소는 1로 채워짐
print(data3)

data4 = np.zeros(3)  # 1 x 3 의 배열로 모든 요소는 0으로 채워짐
print(data4)

data5 = np.identity(3)  # 3 x 3 의 단위 행렬을 생성함
print(data5)            # 단위 행렬 : 행렬의 대각선은 1로 구성되고
                        # 대각은 0으로 채워지며, 대칭인 행렬을 의미함

## 3) 배열 객체
### 배열 객체는 값만 참조함
### 슬라이싱 시, 결과는 동일하게 보이지만
### 리스트의 경우 부분집합만 복사 후 반환
### 배열은 해당사항 없음
list1 = list(range(5))
list2 = list1[0:3]
print(list1)
print(list2)

arr1 = np.arange(5)
arr2 = arr1[0:3]
print(arr1)
print(arr2)

### 2차원 이상의 배열에서 각 요소는 1차원배열로 인식
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr2d
print(arr2d[2])  # [7 8 9]
print(arr2d[0][2])  # 3
print(arr2d[0,2])   # 3

## 4) 배열 전치
t_arr2d = arr2d.T
print(t_arr2d)

## 5) 정렬
arr3 = np.random.randn(8)
arr3
print(arr3)

arr3.sort()
print(arr3)


# 2. Pandas
## 1) 파일 읽기
import pandas as pd

df = pd.read_csv("data/Futures_League/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2016.csv")
print(df)

## 2) 구조 확인
df.info
df.head()
df.tail()

## 3) 파일 쓰기
df.to_csv("result/Futures_League/result.csv", index=False, header=True)
# 만약 경로 내 디렉터리가 존재하지 않으면 아래의 에러가 발생함
# FileNotFoundError: [Errno 2] No such file or directory: 'result/Futures_League/result.csv'
# 디렉터리 생성 코드는 다음과 같다.
import os
if os.path.isdir("result/Futures_League/") is False:
    os.mkdir("result/Futures_League/")
df.to_csv("result/Futures_League/result.csv", index=False, header=True)

## 4) 데이터프레임
data = {'state' : ['Ohio',' Texas','California','Nevada'],
        'year' : [2000, 2001, 2002, 2001],
        'pop' : [1.5, 1.7, 3.6, 2.4]}
data
df2 = pd.DataFrame(data)
df2.info()  # 데이터프레임 정보 확인
print(df2)

df2 = pd.DataFrame(data, columns=['year','state','pop'])  # 컬럼 직접 지정
df2.info()
print(df2)

### index(행 번호) 변경
import numpy as np
df3 = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a','c','d'], columns=['Ohio','Texas','California'])
df3

df3_1 = df3.reindex(['a','b','c','d'])
df3_1

states = ['Texas','utah','California']
df3_1 = df3_1.reindex(columns=states)
df3_1

### 행 또는 열 삭제
df3_1.drop(['Texas', 'utah'], axis=1)

### 행 기준 데이터 인덱싱 및 셀렉팅
df4 = pd.DataFrame(np.arange(16).reshape((4,4)),
                   index=['Ohio','Colorado','Utah','New York'],
                   columns=['one','two','three','four'])
df4
# 특정 조건만 만족하는 데이터 출력
df4[df4['three'] > 5]

### 행 번호 기준 데이터 인덱싱 & 셀렉팅
### 반드시 숫자형 값이 들어와야 함
df4
df4.iloc[[1, 2]]
df4.iloc[np.where(df4.three > 5)]

### lamda 식을 이용한 함수 적용
### - lambda 식 : 1줄형 함수
f = lambda x : x.max() - x.min()

df4
df4.apply(f)
df4.apply(f, axis=1)

### 정렬
df4
df4.sort_index()  # 인덱스를 기준으로 정렬

### 누락값 확인 및 처리
df3_1
df3_1.dropna()  # NA 값인 경우들 제거
df3_1[df3_1.notna()]  # NA 값이 아닌 경우들만 가져옴

df3_1.fillna(0)  # NA 값인 경우 0 으로 채움
df3_1

df3_1.fillna(0, inplace=True) # inplace 옵션 : 새로운 객체를 생성하지 않고, 기존 객체를 수정함
df3_1

### 중복값 처리
df5 = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                    'k2' : [1,1,2,3,3,4,4]})
df5

df5.duplicated()  # 중복이 존재하는 지 boolean 값으로 반환
df5.drop_duplicates()  # 중복값 제거
df5.drop_duplicates(keep='first')  # 처음에 등장하는 값만 유지, 이후의 중복 값은 제거
df5.drop_duplicates(keep='last')  # 마지막에 등장하는 값만 유지, 이전의 중복 값은 제거
df5.drop_duplicates(keep='last', inplace=True)  # inplace 옵션 : 새로운 객체를 생성하지 않고, 기존 객체를 수정함

### 값 치환하기
df5.replace(4, 5)

### 데이터프레임 병합
#### merge
#### - 하나 이상의 키를 기준으로 결합
df6 = pd.DataFrame({'key': ['b','b','a','c','a','a','b'], 'data1': range(7)})
df7 = pd.DataFrame({'key': ['a','b','d'], 'data2': range(3)})

df6
df7

pd.merge(df6, df7, how='inner')  # inner join
pd.merge(df6, df7, how='outer')  # full outer join
pd.merge(df6, df7, how='left')   # left outer join
pd.merge(df6, df7, how='right')  # right outer join

df8 = pd.DataFrame({'lkey': ['b','b','a','c','a','a','b'], 'data1': range(7)})
df9 = pd.DataFrame({'rkey': ['a','b','d'], 'data2': range(3)})

pd.merge(df8, df9, how='inner')
# 동일한 키가 없기 때문에 아래의 에러 발생
# pandas.errors.MergeError: No common columns to perform merge on. Merge options: left_on=None, right_on=None, left_index=False, right_index=False

pd.merge(df8, df9, how='inner', left_on='lkey', right_on='rkey')
# left(right)_on : 왼쪽(오른쪽)의 데이터프레임에서 사용할 키를 지정

#### 인덱스를 이용한 merge
left1 = pd.DataFrame({'key': ['a','b','a','a','b','c'], 'value':range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a','b'])

pd.merge(left1, right1, left_on='key', right_index=True)
# 왼쪽 데이터프레임에서는 'key' 컬럼을, 오른쪽 데이터프레임에서는 인덱스를 키로 사용하겠다는 의미


#### concat
#### - 하나의 축을 따라 붙임
pd.concat([df8, df9])  # 행을 축으로 결함
pd.concat([df8, df9], axis=1)  # 열을 축으로 결합
pd.concat([df8, df9], ignore_index=True)  # 인덱스가 불필요한 경우(처음부터 순차적으로 인덱스를 부여함)

#### 더미변수
#### - 주로 명목형 변수(카테고리 변수)에 대해 플래그와 유사한 이진 값으로 만들어줌
#### - 지시자 역할을 함
df_org = pd.DataFrame({'key' : ['b','b','a','c','a','b'],'data1' : range(6)})
df_dummy = pd.get_dummies(df_org["key"])

df_org
df_dummy

#### 기술통계 계산하기
df = pd.DataFrame([
                        [1.4, np.nan],
                        [7.1, -4.5],
                        [np.nan, np.nan],
                        [0.75,-1.3]
                   ],
                   index=['a','b','c','d'],
                   columns=['one','two'])
print(df)

df.sum() # 컬럼별 총합
df.mean()  # 컬럼별 평균
df.mean(skipna=True)  # NA 값을 제외하지 않고 계산
df.min()  # 컬럼별 최소값
df.max()  # 컬럼별 최대값
df.cumsum()  # 누적합
df.describe()  # 기술통계 : 모든 변수에 대한 기술통계치를 출력함