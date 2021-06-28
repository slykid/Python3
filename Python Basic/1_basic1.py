import numpy as np
import pandas as pd

# 1. 변수 선언
a = 7
print(a)  #  7

type(a)   # int

# 2. 정수
print(3)

print(10 + 5)
print(10 - 5)
print(10 * 5)
print(10 / 5)
print(11 // 5)
print(11 % 5)

# 3. 진수
print(10)
print(0b10)  # 1 x 2 + 0 x 1 = 2
print(0o10)  # 1 x 8 + 0 x 1 = 8
print(0x10)  # 1 x 16 + 0 x 1 = 16

print(int(True))   # 1
print(int(False))  # 0
print(int(98.6))   # 98
print(int('99'))   # 99
print(int('-23'))  # -23
print(4 + 7.0)     # 11.0
print(True + 3)    # 4
print(False + 5.0) # 5.0
print(int('ㅎㅇ'))
# Traceback (most recent call last):
#   File "D:\Program\Anaconda\lib\site-packages\IPython\core\interactiveshell.py", line 3343, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-14-f9de66dc59d8>", line 1, in <module>
#     print(int('ㅎㅇ'))
# ValueError: invalid literal for int() with base 10: 'ㅎㅇ'

# 4. 문자열
print('Snap')
print("Crackle")
print("'Nay,' said the naysayer")

poem = '''There was a Young Lady of Norway,
Who casually sat in a doorway;
When the door squeezed her flat,
She exclaimed, "What of that?"
This courageous Young Lady of Norway'''
print(poem)
poem = 'There was a Young Lady of Norway,
#   File "<ipython-input-17-c52b397313fe>", line 1
#     poem = 'There was a Young Lady of Norway,
#                                              ^
# SyntaxError: EOL while scanning string literal
