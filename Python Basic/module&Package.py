# 1. 패키지
# 1) 상대 경로
# - .. : 부모 디렉터리
# - .  : 현재 디렉터리

# import pkg.fibonacci as fibos
from pkg.fibonacci import Fibonacci as F

import pkg.prints as prts
import pkg.calc as calcs

# 모듈이 클래스형태인 경우
print(F().title)
print(F.fibo(10))
print(F.fibo2(10))

prts.prt1()
prts.prt2()

# 모듈내에 함수를 사용하는 경우
print(calcs.addition(5, 10))
print(calcs.subtract(5, 10))
print(calcs.multiply(5, 10))
print(calcs.multiply(5, 10))
