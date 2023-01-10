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