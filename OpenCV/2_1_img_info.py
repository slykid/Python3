import sys
import cv2

# 입력 영상 불러오기
img1 = cv2.imread('data/opencv/ch01/cat.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/opencv/ch01/cat.bmp', cv2.IMREAD_COLOR)

if img1 is None or img2 is None:
     print("Image load failed!")
     sys.exit()

# 1. 영상 속성 확인
print(type(img1))  # <class 'numpy.ndarray'>
print(img1.shape)  # (480, 640)
print(img1.dtype)  # uint8

if len(img1.shape) == 2:
    print('Image is Grayscale')
elif len(img1.shape) > 2:
    print('Image is TrueColor')

h, w = img1.shape[:2]
print('w x h = {} x {}'.format(w, h))  # w x h = 640 x 480

print(type(img2))  # <class 'numpy.ndarray'>
print(img2.shape)  # (480, 640, 3)
print(img2.dtype)  # uint8

if len(img2.shape) == 2:
    print('Image is Grayscale')
elif len(img2.shape) > 2:
    print('Image is TrueColor')

h, w, c = img2.shape[:3]
print('w x h x c = {} x {} x {}'.format(w, h, c))  # w x h x c = 640 x 480 x 3

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey()
cv2.destroyAllWindows()

# 2. 픽셀값 처리
# - 연산량이 많이 발생할 수 있는 작업이기 떼믄에, 일반적으로는 하지 않는 것이 좋음
# - 특히 동영상의 경우, 연산량이 많아 처리하는 데 더 오래걸림
x = 20
y = 10
img1[y, x] = 0
img2[y, x] = (0, 0, 255)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey()