# 문제. 배열을 입력받아 합으로 0을 만들 수 있는 3개의 요소를 출력하시오
# 입력. [-1, 0, 1, 2, -1, -4]
# 출력.
# [
#    [-1, 0, 1],
#    [-1, -1, 2],
# ]
def solution(input):

    results = []
    input.sort()

    for idx in range(0, len(input) - 2):  # 해당 요소를 담은 리스트의 길이가 3이어야 하므로
        if idx > 0 and input[idx] == input[idx - 1]:
            continue

        left, right = idx + 1, len(input) - 1

        while left < right:
            sum = input[idx] + input[left] + input[right]

            if sum < 0:
                left += 1
            elif sum > 0:
                right -= 1
            else:
                results.append([input[left], input[idx], input[right]])

                while (left < right) and (input[left] == input[left + 1]):
                    left += 1
                while (left > right) and (input[right] == input[right - 1]):
                    right -= 1
                left += 1
                right -= 1

    return results

