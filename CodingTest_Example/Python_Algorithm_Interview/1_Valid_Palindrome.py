# 문제
# 주어진 문자열이 팰린드롬인지 확인해라.
# 대소문자를 구분하지 않으며, 영문자와 숫자만을 대상으로 한다.

# 팰린드롬(Palindrome)
# - 앞 뒤를 뒤집어도 같은 말이 되는 단어나 문장

# [입력]
# "A man, a plan, a canal: Panama" -> True
# "race a car -> False

# [나의 답]
def solution(input: str):
    _input = input.upper()
    _input = _input.replace(",", "").replace(":", "").replace(" ", "")

    _result = None
    if _input.__eq__(_input[::-1]):
        return True
    else:
        return False

print(solution("A man, a plan, a canal: Panama"))
print(solution("race a car"))

# [풀이1 - 리스트 변환]
s = "A man, a plan, a canal: Panama"
def solution(s):
    strs = []
    for char in s:
        if char.isalnum():
            strs.append(char.lower())

    # 펠린드롬 여부 판별
    while len(strs) > 1:
        if strs.pop(0) != strs.pop():
            return False
    return True
print(solution(s))

# [풀이 2 - 덱을 활용한 최적화]
def solution(s: str):
    import collections
    strs: Deque = collections.deque()

    for char in s:
        if char.isalnum():
            strs.append(char.lower())


    while len(strs) > 1:
        if strs.popleft() != strs.pop():
            return False

    return True

print(solution(s))

# [풀이 3 - 슬라이싱 사용]
def solution(s):
    import re

    s = s.lower()
    s = re.sub('[^a-z0-9]', '', s)

    return s == s[::-1]