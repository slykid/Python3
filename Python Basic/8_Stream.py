
# 1. 텍스트 데이터
# 1) write
poem = ''' There was a young lady named Bright, 
Whose speed was far faster than light:
She started one day
In a relative way,
And returned on the previous night'''

fout = open('relativity', 'wt')
print(poem, file=fout, sep='', end='')
fout.close()

fout = open('relativity', 'wt')
size = len(poem)
offset = 0
chunk = 100

while True:
    if offset > len:
        break
    fout.write(poem[offset:offset + chunk])
    offset += chunk
fout.close()

try:
    fout = open('relativity', 'xt')
    fout.write("Test for mode 'x'")
except FileExistsError:
    print('relativity already exists! Check out file name.')

# 2) read(), readline(), readlines()
fin = open('relativity', 'rt')
poem = fin.read()
fin.close()
len(poem)  # 150

poem = ''
fin = open('relativity', 'rt')
chunk = 150
while True:
    fragment = fin.read(chunk)
    if not fragment:
        break
    poem += fragment
fin.close()
len(poem)  # 150

poem = ''
fin = open('relativity', 'rt')
chunk = 150
while True:
    line = fin.readline()
    if not line:
        break
    poem += line
fin.close()
len(poem)  # 150

poem = ''
fin = open('relativity', 'rt')
for line in fin:
    poem += line
fin.close()
len(poem)  # 150

fin = open('relativity', 'rt')
lines = fin.readlines()
fin.close()

print(len(lines), 'lines read')
for line in lines:
    print(line, end='')

# 5 lines read

# There was a young lady named Bright,
# Whose speed was far faster than light:
# She started one day
# In a relative way,
# And returned on the previous night>>>


# 2. 이진 데이터
# 1) write
bin_data = bytes(range(0, 256))
len(bin_data)

fout = open('bfile', 'wb')
fout.write(bin_data)  # 256
fout.close()

fout = open('bfile', 'wb')
size = len(bin_data)
offset = 0
chunk = 100
while True:
    if offset > size:
        break
    fout.write(bin_data[offset:offset+chunk])
    offset += chunk
fout.close()

# 2) read()
fin = open('bfile', 'rb')
bin_data = fin.read()
len(bin_data)  # 256
fin.close()


# 3. with 사용법
with open('relativity', 'wt') as fout:
    fout.write(poem)

# 4. 파일 내 위치 찾기
fin = open('bfile', 'rb')
fin.tell()
fin.seek(255)

bdata = fin.read()
len(bdata)
bdata[0]

# 2. 구조체 파일
# 1) csv
import csv

villains = [
    ['Doctor', 'No'],
    ['Rosa', 'Klebb'],
    ['Mister', 'Big'],
    ['Auric', 'Goldfinger'],
    ['Ernst', 'Blofeld'],
]

with open('villains', 'wt') as fout:
    csvout = csv.writer(fout)
    csvout.writerows(villains)

with open('villains', 'rt') as fin:
    cin = csv.reader(fin)
    villains = [row for row in cin]

print(villains)
