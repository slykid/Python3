# 문제. 한 번의 거래로 낼 수 있는 최대 이익을 산출하라.
# 예시
# 입력: [7, 1, 5, 3, 6, 4]
# 출력: 5

# 내 풀이
import sys


def solution(input):
    _gain = 0
    _buy = 0
    _gap = 0

    for i in input:
        if _buy == 0:
            if i == min(input):
                _buy = i
            else:
                pass
        else:
            if max(abs(i - _buy), _gap) > _gap :
                _gap = abs(i - _buy)
            else:
                pass
    return _gap

# 해답 풀이
def solution2(input):
    benefit = 0
    min_price = sys.maxsize

    for i in input:
        min_price = min(min_price, i)
        benefit = max(benefit, i - min_price)

    return benefit

solution([7, 1, 5, 3, 6, 4])
solution([8, 1, 2, 3, 4, 9, 10])

solution2([7, 1, 5, 3, 6, 4])
solution2([8, 1, 2, 3, 4, 9, 10])