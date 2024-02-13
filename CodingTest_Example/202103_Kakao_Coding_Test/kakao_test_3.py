#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'solution' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY start_time
#  2. INTEGER_ARRAY running_time
#

def solution(start_time, running_time):
    # Write your code here
    sessions = []
    for i in range(len(start_time)):
        sessions.append([start_time[i], running_time[i]])

    count = 1
    idx = 1
    for session in sessions:
        if idx >= len(start_time):
            break
        else:
            if (session[0] + session[1]) <= start_time[idx]:
                count += 1
            idx += 1

    return count
