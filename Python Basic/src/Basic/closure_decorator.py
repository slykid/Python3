import time

# 1. 클로저
# - 함수안의 함수를 결과로 반환할 때의 내부 함수를 의미함
# - 사용 예시: callback, decorator 등
def mul3(n):
    return n * 3

mul3(3)
# 5를 곱하는 함수, 8을 곱하는 함수 등을 만들때도 일일이 생성...?

# 1.1 클래스를 이용하기
class Mul:
    def __init__(self, m):
        self.m = m

    # 이게 클로저
    def mul(self, n):
        return self.m * n

# 1.2 클로저 사용하기
def mul_closure(m):
    def wrapper(n):
        return m * n


# 2. 데코레이터
# - 함수를 꾸며주는 함수
# - @ 를 이용한 어노테이션 방식을 사용
# - 사용 예시: 반복되는 작업을 여러 함수에 적용해야하는 경우
def func1(a, b):
    start_time = time.time()
    print("함수 실행 시작")

    val = a + b

    end_time = time.time()
    print("함수 수행 시간: %.2f 초" % (end_time - start_time))

    return val


if __name__ == "__main__":
    # 1.1 클래스를 이용한 방식
    mul3 = Mul(3)
    mul5 = Mul(5)

    print(mul3.mul(10))
    print(mul5.mul(10))

    # 1.2 클로저를 이용한 방식
    mul3_closure = mul_closure(3)
    mul5_closure = mul_closure(5)

    print(mul3_closure(10))
    print(mul5_closure(10))


    # 2. 데코레이터 예시
    result = func1(1, 2)
    print(result)


