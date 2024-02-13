# 문제. 아군과 적군이 전투를 시작함
# t 는 적군 기지이며, 요소 수 만큼 기지가 존재, 각 요소 값은 적군 전차수 임
# 하나의 기지를 점령하려면, 적군이 배치한 전차 수 보다 1개 더 많게 기지에 보내야함
# 이때, m 개의 기지만큼 점령하려면 필요한 아군 전차수를 출력하시오

# ex1. [3,5,2,9,8], 3, 13
# ex2. [4,2,3,1], 2, 5

def solution(t, m):
    answer = 0

    answer_list = []

    for i in range(m):
        answer_list.append(min(t))
        t.pop(t.index(min(t)))

    answer = sum(answer_list) + m

    return answer