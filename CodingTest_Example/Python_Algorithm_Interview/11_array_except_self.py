# 문제. 배열을 입력받아 output[i] 가 자신을 제외한 나머지 모든 요소의
#      곱셈 결과가 되도록 출력하다
# 입력: [1, 2, 3, 4]
# 출력: [24, 12, 8, 6]

def solution(input):

    result = 0
    for i in range(len(input)):
        