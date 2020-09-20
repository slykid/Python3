# 1. 함수 정의
# 1) 함수 사용
# - 함수를 사용하는 이유 : 반복적 혹은 중복적인 과정을 효율적으로 구현하기 위해서 사용하는 방법
# -  하나의 기능을 하나의 함수로 구현하는 것이 좋음

# 2) 함수 정의
# def function_name(parameter):
#     code

# 3) 함수 사용
# function_naem(parameter)

# 함수 사용 시 선언하는 위치가 중요함
# - 반드시 선언을 먼저 해주고 그 다음 실행해줘야함

# ex1
def hello(world):
    print("Hello", world)

hello("Python")
hello("777")

# ex2
def hello_return(world):
    val = "Hello " + str(world)
    return val

str = hello_return("Python!!")
print(str)

# ex3. 다중리턴
def func_mul(x):
    y1 = x * 100
    y2 = x * 200
    y3 = x * 300

    return y1, y2, y3

val1, val2, val3 = func_mul(300)
print(val1, val2, val3)


# ex4. 데이터 타입반환
def func_mul_list(x):
    y1 = x * 100
    y2 = x * 200
    y3 = x * 300

    return [y1, y2, y3]

list1 = func_mul_list(100)
print(list1)

# *args *kwargs
# - *args : 가변 인자(parameter) / 튜플형식으로 받음

def args_func(*args):
    print(args)
    print(type(args))

args_func("kim")
args_func("kim", "park")

def args_func2(*args):
    for i, v in enumerate(args):
        print(i, v)

args_func2("kim")
args_func2("kim", "park")

# - *kwargs : 키워드 가변인자 / 딕셔너리 형식으로 받음
def kwargs_func(**kwargs):
    print(kwargs)

kwargs_func(name1="kim", name2="park", name3="Lee")

def kwargs_func2(**kwargs):
    for k, v in kwargs.items():
        print("Key: ", k, " Value: ", v)

kwargs_func2(name1="kim", name2="park", name3="Lee")

def example_mul(arg1, arg2, *args, **kwargs):
    print(arg1, arg2, args, kwargs)

example_mul(10, 20)
example_mul(10, 20, 'park', 'kim')
example_mul(10, 20, 'park', 'kim', age1 = 25, age2=30)

# 중첩함수(closure)
def nested_func(num):
    def func_in_func(num):
        print(num)

    print("in func")
    func_in_func(num+1000)

nested_func(100)

# 힌트 사용하기
def func_mul_list(x : int) -> list:
    y1 = x * 100
    y2 = x * 200
    y3 = x * 300

    return [y1, y2, y3]

# 2. Lamda 식 정의
# 1) Lambda
# - 메모리 절약 , 가독성 향상 코드 간결
# - 함수는 객체 생성 -> 리소스(메모리) 할당
# - 람다는 즉시실행(Heap 초기화) -> 메모리 초기화

# - 과용할 경우 가독성이 저하됨


def mul_10(num : int) -> int:
    return num * 10

# 일반 함수 사용법
var_func = mul_10
print(var_func)  # 함수에 대해 직접적으로 값을 넣어 사용하지 않았지만, 객체는 생성됨
#  출력결과 : <function mul_10 at 0x0000022144EDAD30>  <- 객체 생성의 증거
print(type(var_func))  # <class 'function'>
print(var_func(10))

# Lamda 식 사용
lambda_mul_10 = lambda num: num * 10
print(lambda_mul_10(10))

def func_final(x, y, func):
    print(x * y * func(10))
    
func_final(10, 10, lambda_mul_10)

print(func_final(10, 10, lambda x : x * 1000))  # 출력 시 None 이 나오는 이유: 더이상 출력할 것이 없어서 None 이 출력됨 
 


