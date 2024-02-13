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
# The function accepts following parameters:
#  1. 2D_STRING_ARRAY items
#  2. INTEGER orderBy
#  3. INTEGER orderDirection
#  4. INTEGER pageSize
#  5. INTEGER pageNumber
#

def solution(items: list[list[str]], orderBy: int, orderDirection: int, pageSize: int, pageNumber: int):
    # Write your code here

    # 상품정보 정렬
    if orderDirection == 0: _reverse = False
    else: _reverse = True

    items.sort(key= lambda x: x[orderBy], reverse=_reverse)

    # 페이지 할당
    pages = []
    page = []
    for item in items:
        page.append(item[0])
        if len(page) == pageSize:
            pages.append(page)
            page = []

    return pages[pageNumber]
