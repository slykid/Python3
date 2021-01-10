# 1. 클래스
# 1) 클래스 선언
# class class_name:
#     func1,
#     func2,
#     ...

# 속성, 메소드 선언
# - 속성: 클래스 내의 변수
# - 메소드: 클래스 내의 동작 함수

class UserInfo:
    def __init__(self, name): #, height, weight, address):
        self.name= name
        # self.height = height
        # self.weight = weight
        # self.address = address

    def user_info_print(self):
        print("Name: ", self.name)


user1 = UserInfo("kim")
user2 = UserInfo("park")

user1.user_info_print()
user2.user_info_print()

print(id(user1))
print(id(user2))

print(user1.__dict__)
print(user2.__dict__)

# 차이는?
# func1() 의 경우 자신을 호출하는 self 매개변수가 없기 때문에 객체화 했을 때는 호출이 불가하지만,
# 클래스의 입장에서는 자신의 클래스에 소속된 메소드 이므로 호출이 가능함
# 이를 클래스 메소드라고 부름름

#클래스와 인스턴스의 차이
# - 클래스는 추상적인 개념 ( = 붕어빵틀)
# - 인스턴스는 클래스를 이용해 실제 메모리에 로드된 객체 ( = 붕어빵)

# 네임스페이스
# - 각 객체 별로 갖고 있는 고유한 공간
print(user1.name)
print(user2.name)

print("Memory In : ", id(user1))  # id() : 메모리에 저장된 주소값
print("Memory In : ", id(user2))
print(user1.__dict__)
print(user2.__dict__)

# 클래스 변수 : 직접 사용가능, 객체보다 먼저 생성됨
# 인스턴스 변수 : 객체마다 별도로 존재, 인스턴스 생성 후 사용가능함

# self
class SelfTest:
    def func1():
        print("function 1 called")
    def func2(self):
        print("function 2 called")

self_test = SelfTest()
# self_test.func1()  # TypeError: func1() takes 0 positional arguments but 1 was given -> 클래스 메소드
SelfTest.func1()  # function 1 called -> 클래스 메소드 이기 때문에 해당 방법으로 호출해야 사용가능

self_test.func2()  # self 로 인해 메모리에 객체가 생성되었기 때문에 사용이 가능함
SelfTest.func2()  # TypeError: func2() missing 1 required positional argument: 'self'
                  #             -> func2 의 경우 self 를 인자로 받기 때문에 에러가 발생함
SelfTest.func2(self_test)  # function 2 called -> 해당방법으로는 사용이 가능함

print(id(self_test))

# -> 결론 : 인스턴스 메소드의 경우에는 객체를 생성한 후에 호출하고,
#           클래스 메소드의 경우에는 클래스.메소드명() 으로 객체를 호출해야한다.

# 클래스 변수 vs. 인스턴스 변수
class Warehouse:
    # 클래스 변수
    stock_num = 0

    def __init__(self, name):
        self.name = name
        Warehouse.stock_num += 1

    def __del__(self):
        Warehouse.stock_num -= 1

user1 = Warehouse("Kim")
user2 = Warehouse("Park")
user3 = Warehouse("Cho")

print(user1.__dict__)
print(user2.__dict__)
print(user3.__dict__)
print(Warehouse.__dict__)

print(user1.name)
print(user2.name)
print(user3.name)

print(user1.stock_num)  # 클래스 내에 있는 변수이기 때문에, 객체 자신의 네임스페이스가 아닌 클래스의 네임스페이스에 가서 값을 찾음
print(user2.stock_num)
print(user3.stock_num)

print(user1.stock_num)
del user1  # 객체 삭제
print(user2.stock_num)
print(user3.stock_num)
