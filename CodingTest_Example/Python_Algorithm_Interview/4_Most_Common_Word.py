# 문제
# 금지된 단어를 제외한 가장 흔하게 등장하는 단어를 출력하라.
# 대소문자 구분은 없으며, 구두점(마침표, 쉼표) 도 무시한다.

# [입력]
# paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
# banned = ["hit"]

# [출력]
# "ball"

def solution(paragraph: str, banned: list) -> str:
    import re

    input = [x for x in re.sub('[^\w]', " ", paragraph).lower().split() if x not in banned]

    _count = {}
    for word in input:
        if word not in _count.keys():
            _count[word] = 0
        _count[word] += 1

    return max(_count, key=_count.get)

paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]

print(solution(paragraph, banned))
