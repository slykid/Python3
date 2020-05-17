import numpy
# 주석 처리
print(1+2)  # 3

# 라인 유지
alphabet = 'abcdefg' + \
           'hijklmno' + \
           'pqrs'

print(alphabet)

# 조건문
res = True
if res :
    print("TRUE")
else :
    print("FALSE")

# 중첩 조건문
furry = True
small = False

if furry:
    if small:
        print("It's a cat")
    else :
        print("It's a bear")
else:
    if small:
        print("It's a skink")
    else:
        print("It's a human or hairless bear")

# if ~ elif .. else
color = "puce"
if color == "red":
    print("It's tomato")
elif color == "green":
    print("It's green pepper")
else:
    print("I have no idea")

# 조건 우선순위
x = 8
if (x > 5) and (x < 10):
    print(x)


# 반복문
## while
count = 0
while True:
    count += 1
    if count == 7:
        break
    elif count == 5:
        continue
    print(count)

days = ["Mon", "Tue", "Wed"]
fruits = ["apple", "banana", "cherry"]
drinks = ["Coke", "cider", "fanta"]

for day, fruit, drink in zip(days, fruits, drinks):
    print("day : " + day + " fruit : " + fruit + " drink : " + drink )

for idx in range(len(days)):
    print("Today is " + days[idx] + ".")