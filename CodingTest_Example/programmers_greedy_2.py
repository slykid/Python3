# < 문제 >
# 조이스틱으로 알파벳 이름을 완성하세요. 맨 처음엔 A로만 이루어져 있습니다.
# ex) 완성해야 하는 이름이 세 글자면 AAA, 네 글자면 AAAA
#
# 조이스틱을 각 방향으로 움직이면 아래와 같습니다.
#
# ▲ - 다음 알파벳
# ▼ - 이전 알파벳 (A에서 아래쪽으로 이동하면 Z로)
# ◀ - 커서를 왼쪽으로 이동 (첫 번째 위치에서 왼쪽으로 이동하면 마지막 문자에 커서)
# ▶ - 커서를 오른쪽으로 이동
# 예를 들어 아래의 방법으로 "JAZ"를 만들 수 있습니다.
#
# - 첫 번째 위치에서 조이스틱을 위로 9번 조작하여 J를 완성합니다.
# - 조이스틱을 왼쪽으로 1번 조작하여 커서를 마지막 문자 위치로 이동시킵니다.
# - 마지막 위치에서 조이스틱을 아래로 1번 조작하여 Z를 완성합니다.
# 따라서 11번 이동시켜 "JAZ"를 만들 수 있고, 이때가 최소 이동입니다.
# 만들고자 하는 이름 name이 매개변수로 주어질 때, 이름에 대해 조이스틱 조작 횟수의 최솟값을 return 하도록 solution 함수를 만드세요.
#
# 제한 사항
# name은 알파벳 대문자로만 이루어져 있습니다.
# name의 길이는 1 이상 20 이하입니다.

# 입출력 예
# name	    return
# "JEROEN"	56
# "JAN"  	23

# 나의 코드 (81.8점)
def solution(name):
    up_down_cnt = []  # 문자별 조이스틱 위, 아래 이동 횟수
    move_left = 0  # 왼쪽으로 이동한 횟수

    for i in range(len(name)):
        if name[i] == 'A':
            up_down_cnt.append(0)
            if i - 1 == 0:
                move_left += 1
        else:
            _up_cnt = (ord(name[i]) - ord('A'))
            _down_cnt = (ord('Z') - ord(name[i]) + 1)
            if _up_cnt >= _down_cnt:
                up_down_cnt.append(_down_cnt)
            else:
                up_down_cnt.append(_up_cnt)

    return (sum(up_down_cnt) + len(name) - 1 - move_left)

# 정답 코드
def solution(name):
    count = 0
    alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    d = {}
    indexes = []
    current_idx = 0
    n = len(name)

    for i in range(len(alpha)):
        d[alpha[i]] = min(i, 26 - i)

    for i in range(n):
        num = d[name[i]]
        count += num
        if num != 0:
            indexes.append(i)

    while True:
        if len(indexes) == 0:
            break

        min_dist = 99
        min_idx = 0

        for it in indexes:
            min_dist2 = min(abs(it - current_idx), n - abs(it - current_idx))
            if min_dist2 < min_dist:
                min_dist = min_dist2
                min_idx = it

        count += min_dist
        indexes.remove(min_idx)
        current_idx = min_idx

    return count