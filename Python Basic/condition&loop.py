import numpy as np
import pandas as pd

# 조건문 (Condition)
# - if , elif, else 문이 있음
# 조건 명시후 반드시 : 으로 닫아주기
# 코드 블록 : tab

# if 문
if 6 >= 5 :
    print ("It's True")

# 실행 결과 : It's True

# if - else
# - if 조건에 만족하지 않을 경우 실행할 모든 코드를 else 문에서 처리
# - 단 if 와 else 문 사이에는 코드블록 처리되지 않은 어떠한 코드도 올 수 없음
if 6 <= 5:
    print("It's True")
else:
    print("It's False")

# 실행 결과 : It's False

# if - elif - else
# if - else 외에 여러가지의 경우가 발생하는 경우에 사용
a, b, c = 10, 6, 4
if a == 9:
    print("First condition is True")
elif b == 6 and c == 4:
    print("Second one is True")
else :
    print("All conditions are False")

# 실행결과 : Second one is True
# 위의 코드에서 확인할 수 있는 것 처럼 조건에는 조건 연산자(and, or, not) 를 사용하여
# 여러 개의 조건을 연결할 수 있다.

# if 조건문이 bool 타입이 아닌 경우들에서 false 로 간주되는 값들은 다음과 같다.
# None, 0, 0.0, '', [], (), {}, set()
# 위의 경우를 제외한 나머지 값들은 전부 True 로 인식한다.
a = 0
if a :
    print("a is not 0 which means True")
else :
    print("a is 0 which means False")

# 실행결과: a is 0 which means False

# 중첩 조건문 (Nested Condition)
# if 조건문을 중첩해서 사용하는 경우
# 중첩은 깊이로 생각할 수 있으며, 제한은 없다.
# => 코드 블록만 잘지켜주면 됨...
a, b = 1, 2
if a == 1:
    if b == 2:
        print("print")
    else:
        print("b is not 2")

# 실행결과 : print


# 반복문
# 특정 조건에 대한 코드를 반복적으로 수행하는 경우에 사용함
# 기본적으로 while 문 , for 문이 존재함

# while 문
i = 0
while i < 11:
    print("i is ", i)
    i += 1

# for 문
for i in range(10):
    # range() : 어떤 값 미만 까지의 숫자를 순서대로 생성함
    #           기본 시작 값 = 0, 입력된 값미만 까지의 숫자 생성
    #           만약 증감 수를 변경할 경우 마지막 인자에 증감수를 입력하면 됨
    print("i is", i)

for i2 in range(1, 11):
    print("i2 is", i2)

for i3 in range(1, 11, 2):
    print("i3 is", i3)

# Sequence 자료형 반복
# Sequence : 순서가 있는 자료형을 의미하며, 문자열, 리스트, 튜플, 딕셔너리, 집합 등이 속한다.
# iterable return function : range, reversed, enumerate, filter, map, zip

# ex1
names = ["Kim", "Lee", "Cho", "Choi", "Yoo"]
for v1 in names:
    print("You are", v1)

# ex2
my_info = {
    "name":"Kim",
    "age" : 27,
    "city" : "Kimpo"
}

# 기본 값은 키를 호출함
for v2 in my_info:
    print("my_info", v2)

# value 만 이용하는 경우
# item 을 이용하는 경우
for v3 in my_info.values():
    print("my_info", v3)

# item 을 이용하는 경우 k, v 모두 갖고 있으므로 매개 변수도 2개를 사용함
for k, v in my_info.items():
    print("my_info", k, v)

name = "KilhyunKim"
new_name = ""
for n in name:
    if n.isupper():
        new_name += n.lower()
    elif n.islower():
        new_name += n.upper()
print(new_name)

# Continue & Break 문
# Continue 는 해당문구 이하의 내용은 처리하지않고 스킵하는 키워드
# Break 는 해당문구 이하의 내용을 처리하지 않고 반복문을 중지하는 키워드
for i in [14, 6, 4, 0, 129, 26, 45, 10, 16, 100]:
    if i == 10: print("Found!");break
    elif i == 0: continue
    else : print("i is", i)

for i in ["1", 2, True, 4.3, complex(5)]:
    if type(i) is float:  # 타입이 실수형인 경우 패스
        continue
    print("i is", i)    # 실수형 외의 타입은 전부 출력

# for - else 문
# for 반복문 정상적으로 동작하면 반복문 내의 코드를 수행
# 만일 for 반복문 내에 break 문이 없는 경우 else 문의 코드가 실행
for i in [14, 6, 4, 0, 129, 26, 45, 10, 16, 100]:
    if i == 1000: print("Found!")
    elif i == 0: continue
    else : print("i is", i)
else:
    print("Not found 1000")

