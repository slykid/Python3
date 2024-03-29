# # 한글, 숫자 분류기
#
# 어떤 프로그래밍 언어이건 배우게 되면 처음 작성해보는 프로그램이 “Hello, World”를 출력해보는 것인데 컴퓨터 비전 관련 머신러닝의 “Hello, World”는 아마도 MNIST의 숫자 인식기일 것이다. 1999년에 처음 발표된 이 데이터셋은 사람이 쓴 숫자 이미지 7만 개로 구성되어 있으며 (6만 개는 훈련용이고 1만 개는 테스트용) 그동안 비전 관련 알고리즘의 벤치마크로 널리 쓰이고 있다. 아래 그림에서 그 데이터셋의 숫자 이미지를 일부 볼 수 있다.
#
# ![image1.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/55ef65e6-2a2b-42f6-8a77-56da29983bef/image1.png)
#
# < MNIST에서 제공되는 숫자 이미지의 일부 예 - 위키피디아 MNIST_database>
#
# 이 문제의 경우 이미지를 이미 처리해서 훈련용 데이터로 제공해주기 때문에 데이터의 전처리 과정이 필요하지 않다. 하지만 모든 머신러닝 모델 개발에서 중요한 부분은 바로 데이터의 전처리이다.
#
# 이번 문제는 MNIST의 숫자 인식기 문제를 조금 다른 분야로 확장한 한글/숫자 인식기의 개발이다. 모든 MN IST처럼 알파벳 문자를 대상으로 하지는 않고 모두 14개의 한글 문자(‘가’부터 ‘하'까지의 14개의 글자)와 10개의 숫자(MNIST처럼 0부터 9까지의 10개의 숫자)를 대상으로 한다. MNIST의 숫자 인식기 문제와 또 다른 점은 훈련용 데이터들이 이미지 파일로 제공된다는 점으로  이미지 전처리 과정이 필요하다는 점이며 이는 코딩을 필요로 한다. 이미지의 크기는 36 * 36이다.
#
# 모두 2800개의 이미지 데이터가 훈련용으로 제공되며 문자 인식 모델이 빌드된 후에는 이를 제출해야 하는 728개의 테스트 이미지에 적용하여 그 결과를 제출하면 된다.
#
# ### 훈련/테스트 데이터 셋 설명
#
# 이번 문제에 사용된 소문자 알파벳 이미지들을 일부 살펴보면 아래와 같다. 앞서 설명된 것처럼 14개의 한글 문자와 10개의 숫자로 구성되어 있다. 각 이미지들은 모두 36 * 36의 크기로 구성된다.
#
# ![image2.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/aa294cbb-9190-4332-8ca5-345dbff1fad8/image2.png)
#
# data 디렉토리를 열면 아래와 같은 파일들을 볼 수 있다:
#
# * training.csv
# * test.csv
# * Images 폴더: 3528개의 png 파일들이 존재
#
# ##### train.csv의 구성
#
# train.csv의 경우 헤더 라인 하나와 2800개의 이미지 파일 이름으로 구성되어 있으며 하나의 라인은 하나의 훈련용 이미지에 해당하며 각 라인의 첫 번째 필드(filename)는 해당 훈련용 이미지의 파일 이름이 된다. 두 번째 필드(label)는 이미지 파일에 해당하는 글자(0-9, ga-ha)가 된다.
#
# 앞서 이야기했듯이 이미지의 픽셀 정보가 MNIST처럼 훈련용 데이터로 제공되지 않기 때문에 여러분들이 직접 이미지 파일을 읽어서 픽셀 데이터를 만들어내는 전처리 과정을 거쳐야 한다. 이미지 파일들은 images 폴더 안에 존재한다
#
# 여러분이 해야 할 일은 이 파일들을 읽어 들이고 이 데이터를 바탕으로 앞의 필드에 주어진 글자를 예측하는 모델을 만드는 것이다. 예를 들어 이 파일의 처음 5라인을 보면 아래와 같다.
#
# | filename   | label |
# |------------|-------|
# | da_40.png  | da    |
# | ja_141.png | ja    |
# | 9_56.png   | 9     |
# | ba_90.png  | ba    |
#
# label과 그에 해당하는 문자는 다음과 같다.
#
# * 0~9: “0”부터 “9”까지의 숫자에 해당한다.
# * ga: “가"
# * na: “나"
# * da: “다"
# * ra: “라"
# * ma: “마"
# * ba: “바"
# * sa: “사"
# * ah: “아"
# * ja: “자"
# * cha: “차"
# * ca: “카"
# * ta: “타"
# * pa: “파"
# * ha: “하"
#
# ##### test.csv의 구성
#
# 앞서 만든 모델로 풀어야 하는 문제들이 들어있는 파일들이 바로 test.csv이다. 이 파일의 구성은 앞서 train.csv와 비슷하게 하나의 헤더 라인(filename)과 728개의 이미지 파일 라인으로 구성되어있다. 이 두 파일의 차이점은 test.csv에는 이미지에 해당하는 글자가 없이 이미지 파일의 이름만 있다는 점이다. 처음 5줄은 다음과 같다.
#
# | filename |
# |----------|
# | 1.png    |
# | 2.png    |
# | 3.png    |
# | 4.png    |
#
# 여러분이 앞서 훈련용 데이터로 모델을 만든 뒤에 할 일은 여기 있는 이미지 파일들을 읽어서 그걸 모델의 입력으로 주고 나오는 예측값을 얻어내는 것이다. 이를 바탕으로 제출해야 할 파일의 포맷에 대해서 뒤에서 바로 설명한다.
#
# ##### images 폴더
#
# 모델 훈련에 필요한 2800개의 이미지와 나중에 훈련된 모델로 인식해서 인식 결과를 제출해야 하는 테스트 이미지 728개가 존재한다. 앞서 이야기했듯이 이 이미지들은 모두 36 * 36의 크기를 갖는다.
#
#
# # 과제
#
# 채점을 위해 다음을 만족하는 csv 파일을 현재 디렉토리에 `submission.csv`라는 이름으로 저장해야 한다.
#
# 여러분이 제출해야 하는 파일은 test.csv와 같은 수의 라인으로 구성이 되어야 하며 첫 번째 라인(헤더)은 다음과 같은 두 개의 필드로 구성이 되어야 한다:
#
# * filename
# * prediction
#
# 먼저 첫 번째 컬럼으로 들어오는 filename은 test.csv에 들어오는 값들이 그대로 들어와야 한다. 두 번째 컬럼은 여러분이 훈련한 모델에 두 번째 컬럼에 해당하는 이미지 파일을 입력으로 주었을 때 나오는 예측값을 넣어주어야 한다 (즉 앞서 24개의 문자 중의 하나가 되어야 하며 train.csv에 있는 label 필드에 있는 값들을 사용한다).
#
# 예를 들어 test.csv의 처음 다섯 라인이 아래와 같다면
#
# | filename |
# |----------|
# | 1.png    |
# | 2.png    |
# | 3.png    |
# | 4.png    |
#
# 제출하는 파일의 filename 필드는 test.csv에서 사용되었던 것들이 그대로 사용되어야 하며 prediction필드의 값으로는 해당 이미지의 인식 결과가 사용되어야 한다. 앞서 예로 사용한 test.csv에 해당하는 최종 제출 파일은 아래와 같은 형태를 갖추어야 한다.
#
# | filename | prediction |
# |----------|------------|
# | 1.png    | 0          |
# | 2.png    | ga         |
# | 3.png    | 1          |
# | 4.png    | da         |
#
# *주의*
#
# 1. csv 파일의 컬럼은 반드시 filename, prediction 순이어야 한다.
# 2. **index 순서(filename)는 반드시 `test.csv`에 나왔던 filename 순을 따라야 한다**
