from urllib import request as req

url = 'https://www.google.co.kr'
conn = req.urlopen(url)
print(conn)

print(conn.read())
print(conn.status)
print(conn.getheader("Content-Type"))