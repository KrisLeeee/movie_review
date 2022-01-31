# movie_review
영화 리뷰의 긍정 부정 평가

## 사용한 데이터셋
1. DACON 대회 train data, test data
train data 5000개, test data 5000개
(출처 : https://dacon.io/competitions/official/235864/data)

2. 네이버 리뷰 데이터셋
train data 150000개, test data 50000개 -> 5000, 10000개로 sampling하여 진행(google colab RAM 부족)
(출처 : https://raw.githubusercontent.com/e9t/nsmc)

## 목표
0(부정)과 1(긍정)으로 분류된 네이버 영화 리뷰 데이터를 분석하고 딥러닝 모델을 통해 학습시킨 후
새로운 리뷰 데이터를 입력했을 때 리뷰의 긍정과 부정을 분류할 수 있도록 설계한다.

## 사용한 모델
keras 패키지의 라이브러리 사용
1. DNN
2. LSTM(RNN)

## 과절
1. 데이터 분석
  - 긍정과 부정 리뷰 각각의 개수, 리뷰의 길이, 단어 수 비교
  - 빈도 수가 높은 단어 상위 10개

2. 토큰화
  - konlpy의 Okt 이용
  - 명사, 부사, 형용사,동사 : 긍정과 부정의 판별에 의미가 있는 형태소만 출력

3. 모뎋 설계
  - Logistic Regression (skit learn library)
  - Soft Vector Machine (skit learn library)
  - DNN (tenserflow keras)
  - RNN by LSTM (tensorflow keras)
