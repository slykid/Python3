# 1. 바이트, 바이트 배열 사용법
input = [1, 2, 3, 255]

i_bytes = bytes(input)
print(i_bytes)

array_bytes = bytearray(input)
array_bytes


# 2. 값 수정 여부 확인
i_bytes[1] = 127

array_bytes[1] = 127
array_bytes


# 3. 값의 표현 범위 확인
i_bytes = bytes(range(0, 256))
array_bytes = bytearray(range(0, 256))

i_bytes
array_bytes

import struct

valid_png_header = b'\x89PNG\r\n\x1a\n'
data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' + \
       b'\x00\x00\x00\x9a\x00\x00\x00\x8d\x08\x02\x00\x00\x00\xc0'

if data[:8] == valid_png_header:
    width, height = struct.unpack('>LL', data[16:24])
    print('Valid PNG, width', width, 'height', height)
else:
    print('Not a valid PNG')