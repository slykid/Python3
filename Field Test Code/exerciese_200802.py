# 아래 문제를 실행해보고 print 문에 대한 실행 결과가 어떤 자료 구조인지 주석으로 남겨주세요

# Q1
character = ["a", "b", "c"]
print(character)

# Q2
number = tuple(range(1,5))
print(number)

# Q3
data3 = {}
for letter in character:
    data3[letter] = {}
    for num in number:
        data3[letter]["num"+str(num)] = num
print(data3)

# Q4
word = 'letters'
letters_count = {letter: word.count(letter) for letter in set(word)}
print(letters_count)

# Q5
data4 = 1,2,3
print(data4)

# Q6
data5 = [x+1 for x in range(5)]
print(data5)

# Q7
rows = range(1, 4)
cols = range(1, 3)
cells = [(row, col) for row in rows for col in cols]
print(cells)
print(cells[0])

# Q8
data6 = {x+1 for x in range(5)}
print(data6)