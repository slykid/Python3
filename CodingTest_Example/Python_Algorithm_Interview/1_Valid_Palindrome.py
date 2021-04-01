# 문제
# 주어진 문자열이 팰린드롬인지 확인해라.
# 대소문자를 구분하지 않으며, 영문자와 숫자만을 대상으로 한다.

# 팰린드롬(Palindrome)
# - 앞 뒤를 뒤집어도 같은 말이 되는 단어나 문장

# [입력]
# "A man, a plan, a canal: Panama" -> True
# "race a car -> False
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