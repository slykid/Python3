# 문제. 높이를 입력받아 비 온 후 얼마나 많은 물이 쌓일 수 있는지 계산하라.
# 입력: [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
# 출력: 6
def solution(input):
    if not input:
        return 0

    volume = 0

    left, right = 0, len(input) - 1
    left_max, right_max = input[left], input[right]

    while left < right:
        left_max, right_max = max(input[left], left_max), max(input[right], right_max)

        if left_max <= right_max:
            volume += left_max - input[left]
            left += 1
        else:
            volume += right_max - input[right]
            right -= 1

    return volume

solution([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])