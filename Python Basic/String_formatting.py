# 1. 옛 스타일 포멧팅 (%)
print("1. 정수")
print('%s' % 42)
print('%d' % 42)
print('%x' % 42)
print('%o' % 42)
print()

print("2. 부동소수점수")
print('%s' % 10.8)
print('%f' % 10.8)
print('%e' % 10.8)
print('%g' % 10.8)
print()

print("3. 문자열 + 정수")
actor = "Richard Gere"
cat = "Chester"
weight = 28

print("My wife's favorite actor is %s" % actor)
print("Our cat %s weights %s pound" % (cat, weight))
print()

print("4. 문자열 길이 조정")
n = 100
f = 10.8
s = "String cheese"

print("%d %f %s" % (n, f, s))
print("%10d %10f %10s" % (n, f, s))

# 2. 새로운 스타일 포맷팅: {} & format()
print("1. {} 사용법")
n = 100
f = 10.8
s = "String cheese"
print("{} {} {}".format(n, f, s))

print("2. {} 사용 시 순서 정하기")
n = 100
f = 10.8
s = "String cheese"
print("{2} {0} {1}".format(n, f, s))

print("3. 자료구조를 활용한 포맷팅")
dict_a = {'n': 100, 'f': 10.8, 's': "String cheese"}
print("{0[n]} {0[f]} {0[s]} {1}".format(dict_a, "other"))

print("4. 문자열 길이 조절하기")
n = 100
f = 10.8
s = "String cheese"
print("{0:d} {1:f} {2:s}".format(n, f, s))

print("5. 문자열 길이 지정")
n = 100
f = 10.8
s = "String cheese"
print("{0:>10d} {1:^10f} {2:<10s}".format(n, f, s))
