# 슈퍼 바이러스
import sys

K, P, N = map(int, sys.stdin.readline().split())

def func(P, N):
  if N == 1:
    return P

  elif (N % 2 == 0):
    result = func(P, N/2)
    return (result * result) % 1000000007

  else:
    result = func(P, (N-1)/2)
    return (result * result * P) % 1000000007

result = func(P, 10 * N) * K

print(result % 1000000007)