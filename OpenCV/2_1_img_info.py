import sys
import cv2

# 입력 영상 불러오기
img1 = cv2.imread('data/opencv/ch01/cat.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/opencv/ch01/cat.bmp', cv2.IMREAD_COLOR)

if img1 is None or img2 is None:
     print("Image load failed!")
     sys.exit()

print(type(img1))
print(img1.shape)
print(img1.dtype)

if len(img1.shape) == 2:
    print('Image is Grayscale')
elif len(img1.shape) > 2:
    print('Image is TrueColor')

h, w = img1.shape[:2]
print('w x h = {} x {}'.format(w, h))

print(img2.shape)
print(img2.dtype)

if len(img2.shape) == 2:
    print('Image is Grayscale')
elif len(img2.shape) > 2:
    print('Image is TrueColor')

h, w, c = img2.shape[:3]
print('w x h x c = {} x {} x {}'.format(w, h, c))

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey()