# 문제. 대학 재학생(1)과 신입생(0) 이 일렬로 서있음, 1개 그룹을 만들건데 재학생은 k명이 포함되야하며,
# 좌우로 인접한 경우만 허용함 이때, 만들 수 있는 그룹 수를 반환하시오
# 그룹이 없을 경우, 0을 반환

# ex1. [0, 1, 0, 0], 1, 6
# ex2. [0, 1, 0, 0, 1, 1, 0], 2, 8
# ex3. [0, 1, 0], 2, 0

def solution(student, k):
    answer = -1
    if sum(student) == 0 or sum(student) < k:
        answer = 0
    else:
        idx = 0
        answer_list = []
        length = k

        for length in range(k, len(student) + 1):
            idx = 0

            for i in range(len(student) - k + 1):
                if sum(student[idx:idx + length]) == k:
                    if student[idx:idx + length] in answer_list:
                        continue
                    else:
                        answer_list.append(student[idx:idx + length])
                idx += 1

        answer = len(answer_list)
    return answer