# Section05-3
# 파이썬 흐름제어(제어문)
# 제어문 관련 퀴즈(정답은 영상)

# 1 ~ 5 문제 if 구문 사용
# 1. 아래 딕셔너리에서 '가을'에 해당하는 과일을 출력하세요.
q1 =  {"봄": "딸기", "여름": "토마토", "가을": "사과"}
for key, value in q1.items():
    if key.__eq__("가을"):
        print(key+"의 과일: ", value)

# 2. 아래 딕셔너리에서 '사과'가 포함되었는지 확인하세요.
q2 =  {"봄": "딸기", "여름": "토마토", "가을": "사과"}
for key, value in q2.items():
    if key.__eq__("사과") or value.__eq__("사과"):
        print("Found!", key, value);break
else:
    print("Not Found...")

# 3. 다음 점수 구간에 맞게 학점을 출력하세요.
# 81 ~ 100 : A학점
# 61 ~ 80 :  B학점
# 41 ~ 60 :  C학점
# 21 ~ 40 :  D학점
#  0 ~ 20 :  E학점
score = 75
if score >= 81:
    print("A grade")
elif score >= 61:
    print("B grade")
elif score >= 41:
    print("C grade")
elif score >= 21:
    print("D grade")
else:
    print("E grade")


# 4. 다음 세 개의 숫자 중 가장 큰수를 출력하세요.(if문 사용) : 12, 6, 18
a, b, c = 12, 6, 18
maxNum = a
if a < b:
    maxNum = b
elif b < c:
    maxNum = c

# 5. 다음 주민등록 번호에서 7자리 숫자를 사용해서 남자, 여자를 판별하세요. (1,3 : 남자, 2,4 : 여자)
idNum = "931008-1575429"
if int(idNum.split("-")[1][0]) == 1 or int(idNum.split("-")[1][0]) == 3:
    print("남자")
elif int(idNum.split("-")[1][0]) == 2 or int(idNum.split("-")[1][0]) == 4:
    print("여자")


# 6 ~ 10 반복문 사용(while 또는 for)

# 6. 다음 리스트 중에서 '정' 글자를 제외하고 출력하세요.
q3 = ["갑", "을", "병", "정"]
for l in q3:
    if l.__eq__("정"):
        continue
    print(l, end="")

# 7. 1부터 100까지 자연수 중 '홀수'만 한 라인으로 출력 하세요.
for i in range(1,100):
    if i % 2 == 1:
        print(i, end=" ")

# 8. 아래 리스트 항목 중에서 5글자 이상의 단어만 출력하세요.
q4 = ["nice", "study", "python", "anaconda", "!"]
for l in q4:
    if len(l) >= 5:
        print(l, end=" ")

# 9. 아래 리스트 항목 중에서 소문자만 출력하세요.
q5 = ["A", "b", "c", "D", "e", "F", "G", "h"]
for l in q5:
    if l.islower():
        print(l, end=" ")

# 10. 아래 리스트 항목 중에서 소문자는 대문자로 대문자는 소문자로 출력하세요.
q6 = ["A", "b", "c", "D", "e", "F", "G", "h"]
for l in q6:
    if l.isupper():
        print(l.lower(), end=" ")
    elif l.islower():
        print(l.upper(), end=" ")