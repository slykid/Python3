# 문제. 덧셈하여 타겟을 만들 수 있는 배열의 두 숫자의 인덱스를 반환하시오.
# 입력: nums = [2, 7, 11, 15] / target = 9
# 출력 [0, 1]

def solution(input: list, target: int):

    for i, num in enumerate(input):
        if (target - num) in input:
            return [input.index(num), input.index(target - num)]

solution([2, 7, 11, 15], 9)
solution([2, 7, 11, 15], 22)
