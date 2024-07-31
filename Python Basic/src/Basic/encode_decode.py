# 1. 인코딩-디코딩
# - 인코딩: 문자열 to 바이트로 변환
# - 디코딩: 바이트 to 문자열로 변환

# 1.1 인코딩
a = "Life is too short!"
type(a)

print(a)


b = a.encode("UTF-8")
type(b)

print(b)

a = '한글'
a.encode("UTF-8")

# 1.2 디코딩
a = '한글'
b = a.encode("UTF-8")
print(b)

b.decode("UTF-8")
b.decode("ASCII")  # UnicodeDecodeError: 'ascii' codec can't decode byte 0xed in position 0: ordinal not in range(128)

print("Learning: ", "\U0001F40D")  # Learning:  🐍