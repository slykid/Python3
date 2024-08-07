# 징검 다리
import sys

cnt = int(sys.stdin.readline())
stones = list(map(int, sys.stdin.readline().split()))

dp = [1] * cnt

for i in range(1, cnt):
    pass_cnt = 0
    for j in range(i):
        if stones[j] < stones[i]:
            pass_cnt = max(pass_cnt, dp[j])

    dp[i] = pass_cnt + 1

print(max(dp))