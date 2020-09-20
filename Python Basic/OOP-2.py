# 1. 클래스 상속
# - 부모클래스(슈퍼클래스) 가 갖고 있는 모든 속성 및 메소드을 자식클래스(서브클래스) 에게 물려주는 것
# - 코드의 생산성, 가독성을 높이기 위한 방법임
# - 파이썬의 경우 다중상속을 허용함

# 상속은 "하위 클래스명(상위 클래스명) :" 과 같이 사용한다.

class Car:
    """Parent Class"""

    def __init__(self, type, color):
        self.type = type
        self.color = color

    def show(self):
        return 'Car Class "Show Method!!"'

class BMW(Car):
    """Sub Class"""

    def __init__(self, car_name, type, color):
        super().__init__(type, color)
        self.car_name = car_name

    def show_model(self) -> None:
        return 'Your Car Name : %s' % self.car_name


class Benz(Car):
    """Sub Class"""

    def __init__(self, car_name, type, color):
        super().__init__(type, color)
        self.car_name = car_name

    def show_model(self) -> None:
        return 'Your Car Name : %s' % self.car_name

    def show(self):
        print(super().show())
        return 'Car Info : %s %s %s' % (self.car_name, self.type, self.color)


model1 = BMW('520d', 'sedan', 'red')
print(model1.color)  # 부모의 color
print(model1.type)  # 부모의 type
print(model1.car_name)  # 자식의 car_name
print(model1.show())  # 부모의 show() 메소드
print(model1.show_model())  # 자식의 show_model() 메소드
print(model1.__dict__)

# 메소드 오버라이딩
# - 부모에 존재하는 메소드를 사용에 맞게 변경하여 사용하는 것
model2 = Benz("220d", "suv", "black")
print(model2.show())

# 부모 메소드 직접호출
model3 = Benz("350s", "sedan", "covelt")
print(model3.show())  # 부모에 있는 메소드 사용 시, super().메소드명() 으로 호출해줄 수 있다.

# 상속 정보 확인
print(BMW.mro())  # mro() : 상속받은 정보를 보여줌 (왼쪽에서 오른쪽으로 읽어가면됨)
print(Benz.mro())

# 다중 상속
class x():
    pass
class y():
    pass
class z():
    pass

class A(x, y):
    pass

class B(y, z):
    pass

class M(B, A, z):
    pass

print(M.mro())  # 너무나 많은 다중 상속은 코드 가독성이 떨어짐!
