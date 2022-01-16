import os

# 1. open()
fout = open('oops.txt', 'wt')
print("Oops, I create a file. ", file=fout)
fout.close()

# 2. exists()
os.path.exists('oops.txt')
os.path.exists('./oops.txt')
os.path.exists('waffles')
os.path.exists('.')
os.path.exists('..')

# 3. isfile(), isdir(), isabs()
name = 'oops.txt'
print(os.path.isfile(name))
print(os.path.isdir(name))
print(os.path.isabs('name'))
print(os.path.isabs('/big/fake/name'))
print(os.path.isabs('big/fake/name/without/a/leading/slash'))

# 4. mkdir()
os.mkdir('poems')
os.path.exists('poems')

# 5. rmdir()
os.rmdir('poems')
os.path.exists('poems')

# 6. listdir()
os.listdir('poems')

# 7. chdir()
os.chdir('poem')
os.listdir('.')

# 8. subprocess
import subprocess
result = subprocess.getoutput('dir')
print(result)

# 9. datetime
from datetime import date

halloween = date(2022, 10, 31)
print(halloween)

print(halloween.day)
print(halloween.month)
print(halloween.year)

now = date.today()
print(now)

from datetime import timedelta

one_day = timedelta(days=1)
tomorrow = now + one_day
print(tomorrow)

print(now + 17 * one_day)


import time

now = time.time()
print(now)

time.localtime(now)
time.gmtime(now)

tm = time.localtime(now)
time.mktime(tm)