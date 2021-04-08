import re

# 정규표현식 사용법1
result = re.match('You', 'Young Frankenstein')
print(result)

# 패턴 컴파일
pattern = re.compile("You")
result = re.match(pattern, 'Young Frankenstein')
print(result)

result = re.match("Frank", 'Young Frankenstein')
print(result)

result = re.search("Frank", 'Young Frankenstein')
print(result)

result = re.findall('n', 'Young Frankenstein')
print(result)

result = re.split('n', 'Young Frankenstein')
print(result)

result = re.sub('n', 'N', 'Young Frankenstein')
print(result)
