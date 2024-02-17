# 1. Auto Encoder
- 인코더(Encoder)와 디코더(Decoder)를 통해 압축과 해제를 실행
  - 인코더는 입력 정보를 최대한 보존하도록 손실 압축을 수행
  - 디코더는 중간 결과의 정보를 입력과 같아지도록 압축 해제(복원)을 수행

- 위의 2가지 과정을 반복하며, 정확한 특징을 추출하는 방법을 학습함

## 1) Encoder 
- 복원에 필요한 정보를 중심으로 손실압축을 수행
- 복원에 필요없는 정보는 버려짐<br>
  ex. 사람 얼굴은 눈이 2개다

- 데이터의 선택과 압축이 발생함(차원에 따라 압축정도를 결정)
- Encoder 의 결과(Bottleneck)는 입력에 대해 필요한 특징들만 추출된 Feature Vector가 된다.<br>
  → 인코더에 입력을 통과시키는 과정 = Embedding

## 2) Decoder
- Encoder 의 결과를 바탕으로 최대한 입력과 비슷하게 복원함
- 뻔한 정보가 없어도 알 수 있기 때문에 복원이 가능함

# 2. Hidden Representation
- DNN에 입력 데이터를 통과 시키는 과정이 인코더를 통과하는 것과 유사함
  - DNN을 구성하는 각 레이어의 결과물을 Hidden Vector 라고 부른다.
- Hidden Vector 는 입력 샘플의 feature 를 담고 있음 
  
- 하지만, Hidden Vector의 해석은 feature vector 해석보다 어려움
- 단, 비슷한 특징을 가진 샘플들의 경우 비슷한 hidden vector를 갖는다.


