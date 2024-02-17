# 문제. 가장 긴 펠린드롬 부분 문자열을 출력하시오.
# Ex1.
# 입력: babad
# 출력: bab

# Ex2.
# 입력: cbbd
# 출력: bb
def solution(input:str):
    import re

    def expand(left, right):
        while left >= 0 and right < len(input) and input[left] == input[right]:
            left -= 1
            right += 1
        return input[left + 1:right]

    _string = re.sub('[^A-Za-z0-9]', '', input).lower()

    result = ''
    if (len(_string) < 2) or (_string == _string[::-1]):
        result = _string
    else:
        for i in range(len(_string) - 1):
            result = max(result, expand(i, i + 2), expand(i, i + 1), key=len)

    return result

solution("babad")
solution("adgsbsbsllfi")