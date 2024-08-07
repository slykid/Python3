# 문제
# 문자열을 뒤집는 함수를 작성하시오.
# 입력값은 문자 배열이며, 리턴 없이 리스트 내부를 직접 조작하시오.

# [입력]  ["h", "e", "l", "l", "o"]
# [출력]  ["o", "l", "l", "e", "h"]

def reverse_string(input: list):
    print(input[::-1])

reverse_string(["h", "e", "l", "l", "o"])
reverse_string(["H", "a", "n", "n", "a", "h"])