import os
import sys
import cv2
import glob

file_list = glob.glob("data/opencv/ch01/images/*.jpg")

for filename in file_list:
    print(filename)

# 전체 화면으로 출력되도록 출력
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Windows 용

# 슬라이드 생성
cnt = len(file_list)
idx = 0

while True:
    img = cv2.imread(file_list[idx])
    cv2.imshow('image', img)

    if cv2.waitKey(1000) == 27:
        break

    idx += 1
    idx = idx % 5

cv2.destroyAllWindows()