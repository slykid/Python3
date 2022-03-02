# 문제. n 개의 페어를 이용한 min(a, b)의 합으로 만들 수 있는 가장 큰 수를 출력하라.
# 입력: [1, 4, 3, 2]
# 출력: 4

def partition(input):

    input.sort()
    return sum(input[::2])

partition([1, 4, 3, 2])
partition([1, 4, 5, 3, 2, 6])


