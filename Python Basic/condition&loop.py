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
