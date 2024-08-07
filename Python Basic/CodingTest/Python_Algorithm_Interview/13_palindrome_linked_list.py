# 문제. 연결 리스트가 팰린드롬 구조인지 확인하시오.
# 예시.
# 입력 1. 1 -> 2 (false)
# 입력 2. 1 -> 2 -> 2 -> 1 (true)

def solution(input):
     q = []

     if not input:
         return True

    node = input

    while node is not None:
        q.append(input.val)
        node.next

    while len(q) > 1:
        if q.pop(0) != q.pop():
            return False

    return True







