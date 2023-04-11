import sys
import cv2
import numpy as np

src = cv2.imread('data/opencv/ch02/airplane.bmp', cv2.IMREAD_COLOR)
mask = cv2.imread('data/opencv/ch02/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('data/opencv/ch02/field.bmp', cv2.IMREAD_COLOR)

# dst 이미지 내에 src 이미지 중 비행기 부분만 복사해서 붙여넣는 경우
cv2.copyTo(src, mask, dst)  # copyTo src 에서 mask 에 해당하는 부분만 복사, 붙여넣기

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()

# copyTo() 사용 시, dst 가 없는 경우
dst = cv2.copyTo(src, mask)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()


import sys
import cv2
import numpy as np

src = cv2.imread('data/opencv/ch02/airplane.bmp', cv2.IMREAD_COLOR)
mask = cv2.imread('data/opencv/ch02/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('data/opencv/ch02/field.bmp', cv2.IMREAD_COLOR)

# numpy를 이용한 방법(Boolean Indexing)
dst[mask > 0] = src[mask > 0]

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()

# 마스크 영상을 이용한 영상 합성
import sys
import cv2
import numpy as np

src = cv2.imread('data/opencv/ch02/opencv-logo-white.png', cv2.IMREAD_UNCHANGED) # 4개 채널의 영상
# mask = cv2.imread('data/opencv/ch02/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
mask = src[:, :, -1]
src = src[:, :, 0:3]
dst = cv2.imread('data/opencv/ch02/field.bmp', cv2.IMREAD_COLOR)

# 입력 영상들을 위의 내용과 같이 설정할 경우 크기가 맞지 않아서 이후 copyTo 연산이 제대로 동작하지 않음
# 따라서 연산 전, 합성 대상(dst) 영상에 대한 높이를 별도로 지정해야 됨
h, w = src.shape[:2] # 소스 영상의 높이, 너비와 동일하게 설정함

crop = dst[0:h, 0:w] # src 영상에 대한 높이, 길이만큼 dst 영상에서 분할함

# cv2.copyTo(src, mask, dst)
cv2.copyTo(src, mask, crop)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()