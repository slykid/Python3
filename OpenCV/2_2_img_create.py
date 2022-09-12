import sys
import cv2
import numpy as np

# 1. 지정한 크기로 영상 생성하기
img1 = np.empty((240, 320), dtype=np.uint8)      # grayscale
img2 = np.zeros((240, 320, 3), dtype=np.uint8)
img3 = np.ones((240, 320, 3), dtype=np.uint8) * 255
img4 = np.full((240, 320), fill_value=128, dtype=np.uint8)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)
cv2.waitKey()
cv2.destroyAllWindows()

# 2. 영상 참조 및 복사
img1 = cv2.imread('data/opencv/ch02/HappyFish.jpg')

# - 아래 2개의 차이는?
#   대입연산자 사용 시, 값의 주소만 대입한 것일 뿐 실제 데이터는 img1 과 공유함
#   copy() 메소드 사용 시, 새로운 데이터를 하나 생성해서 대입함
img2 = img1
img3 = img1.copy()  # ndarray 에서 지원해주는 함수

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey()
cv2.destroyAllWindows()

# 차이 확인
img1[:, :] = (0, 255, 255)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey()
cv2.destroyAllWindows()

# 3. 부분 영상 추출
