# Level 3. 성적 평균
import sys

N, K = map(int, sys.stdin.readline().split())
score = [int(x) for x in sys.stdin.readline().split()]

for i in range(K):
    start, end = map(int, sys.stdin.readline().split())

    # print(sum(score[start-1:end]))
    # print(len(score[start-1:end]))

    avg_score = round(sum(score[start - 1:end]) / len(score[start - 1:end]), 2)

    print("{:.2f}".format(avg_score))
    avg_score = 0