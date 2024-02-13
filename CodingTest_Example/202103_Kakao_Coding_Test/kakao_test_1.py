#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'solution' function below.
#
# The function is expected to return a STRING_ARRAY.
# The function accepts 2D_STRING_ARRAY arr as parameter.
#

def solution(arr):
    # Write your code here
    point = {}

    for row in arr:
        # _list = row.split()
        # 키 등록
        if row[0] not in point.keys():
            point[row[0]] = 0
        if row[1] not in point.keys():
            point[row[1]] = 0

        # 포인트 계산
        point[row[0]] -= int(row[2])
        point[row[1]] += int(row[2])

    # 최소 잔여 포인트 찾기
    min_point = -1
    result = []
    for k, v in point.items():
        if v < 0 and v <= min_point:
            if len(result) == 0 or v == min_point:
                result.append(k)
            else:
                result.pop()
                result.append(k)

            min_point = v

    if len(result) == 0:
        result.append("None")
    else:
        result.sort()

    return result


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr_rows = int(input().strip())
    arr_columns = int(input().strip())

    arr = []

    for _ in range(arr_rows):
        arr.append(input().rstrip().split())

    result = solution(arr)

    fptr.write('\n'.join(result))
    fptr.write('\n')

    fptr.close()
