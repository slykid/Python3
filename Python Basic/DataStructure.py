import pandas as pd
import numpy as np

# 1. 리스트
# 1) 정의
a = list()
b = []
c = [1,2,3,4]
d = [10, 100, "pen", "apple"]
e = [10, 100, ["pen", "apple"]]

print(a)
print(b)
print(c)
print(d)
print(e)

## 인덱싱 & 슬라이싱
print(c[2])        # 인덱싱: 해당 순서에 있는 값을 출력
print(c[1:3])      # 슬라이싱 : 1이상 3미만에 해당하는 인덱스의 요소를 출력
print(c[-3:-1])    # 음수(- 부호)는 리버싱을 의미하며, 역순으로 출력함
print(e[2][0])     # 리스트의 요소가 리스트인 경우 리스트 안의 리스트에 있는 값을 출력할 시 다음과 같음.
print(d[::-1])

# 2) 사용 함수
## append
marxes = ['Grouncho', 'Chico', 'Harpo']
marxes.append('Zepo')
print(marxes)

a.append(1)
print(a)
a.append(2)
print(a)
a.append(6)
a.append(5)

## sort & reverse
print(a)
a.sort()
print(a)
a.reverse()
print(a)

## insert
marxes.insert(3, 'Harpo')
print(marxes)

a.insert(2,7)
print(a)

## remove
a.remove(2)  # 해당 값을 지우는 것임
print(a)
a.insert(3, 2)
print(a)
del a[2]    # del 과의 차이점: 인덱스의 값을 지우는가, 직접 입력된 값을 지우는가
print(a)
a.insert(2,7)
print(a)

## pop
a.pop()     # Stack 과 관련된 함수로 마지막 값을 출력하고 제거함
print(a)    # 맨 처음의 값까지 pop하는 경우 인덱스 에러가 발생함
a.append(1)
print(a)
b = a

marxes = ['Grouncho', 'Chico', 'Zepo', 'Harpo']
print(marxes)
marxes.pop()
print(marxes)

## index
marxes.index('Chico')

## in
print('Chico' in marxes)
print('Chick' in marxes)

## count
marxes = ['Grouncho', 'Chico', 'Harpo', 'Zepo', 'Harpo']
marxes.count('Harpo')

## sort(), sorted()
marxes = ['Grouncho', 'Chico', 'Zepo', 'Harpo']
print(sorted(marxes))
marxes.sort()
print(marxes)

##len()
marxes = ['Grouncho', 'Chico', 'Zepo', 'Harpo']
len(marxes)

a = [1,2,3]
print(a)

b = a
print(b)

a[0] = 'hello'
print(a)
print(b)


for i in range(0, len(b)):
    a.pop()
a.pop()     # IndexError: pop from empty list 발생

## extend
a = [1,2,5,6]
b = [90, 100]
a.extend(b) # 리스트의 값 자체만을 추가함
print(a)
a.append(b) # append와의 차이 : append는 리스트 자체를 하나의 요소로서 추가함
print(a)
a.pop()

print(a)
del a[2]
print(a)

# 2. 튜플
tuple1 = ()
tuple2 = 1,2,3

print(tuple1)
print(tuple2)

a = (1,2)
del a[2]    # TypeError: 'tuple' object doesn't support item deletion 발생

password = 'swordfish'
icecream = 'fruits'
password, icecream = icecream, password

print(password)
print(icecream)

# 1) 사용 함수
## in
print(2 in a)   # 해당 값이 튜플에 존재하는 지 확인

## index
print(a.index(2))   # 해당 값의 인덱스를 출력

## count
a = (1,2,1,3,5)
print(a.count(1))   # 해당 값의 개수를 출력

# 3. 딕셔너리(사전)
a = {"name":"Kilhyun" , "HP":"010-1234-5678", "birth":"2015-12-01"}
b = {0: "hello", 1:"World"}
c = {"arr":[1,2,3,4,5]}

## 출력
a["name"]
print(a.get("name"))
print(a.get("address")) # 키에 대한 값이 없는 경우 None을 반환, 자료에 대한 조회시 이용하면 좋음
print(c["arr"][2])

## 데이터 추가
a["address"] = "Seoul"
print(a.get("address"))

a["rank"] = [1, 3, 6]
a["rank2"] = (1, 5, 7)
print(a)

## Keys, Values, Items
### Item : Key-Value 쌍을 의미
a = {"name":"Kilhyun" , "HP":"010-1234-5678", "birth":"2015-12-01"}
print(a.keys())     # 딕셔너리에 존재하는 모든 Key 를 출력함
print(a.values())   # 딕셔너리에 존재하는 모든 Value 를 출력함
print(a.items())    # 딕셔너리에 존재하는 모든 Item(Key-Value 쌍, 튜플형식) 를 출력함