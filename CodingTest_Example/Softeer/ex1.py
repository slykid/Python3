# A+B
import sys

input = sys.stdin.readline()

T = int(input())
for i in range(T):
    a, b = map(int, input().split())
    print("Case #" + str(i+1) + ": " + str(a+b))