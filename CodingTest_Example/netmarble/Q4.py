# 문제. 자연수 배열 numbers 를 입력받아 과반수 이상의 숫자를 출력하시오
# 숫자가 없는 경우 -1 을 출력하시오
# 과반수란 입력 배열 길이 / 2 를 초과한 경우를 의미함

# ex1. [6, 1, 6, 6, 7, 6, 6, 7], 6
# ex2. [6, 1, 6, 6, 7, 5, 6, 7], -1

def solution(numbers):
    answer = 0

    num_dict = {}

    for num in numbers:
        if num not in num_dict.keys():
            num_dict[num] = 1
        else:
            num_dict[num] += 1

    max_val_key = 0
    max_val = 0
    for key, value in num_dict.items():
        if value > max_val:
            max_val_key = key
            max_val = value

    if max_val > len(numbers) / 2:
        answer = max_val_key
    else:
        answer = -1

    return answer