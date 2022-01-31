# movie_review
영화 리뷰의 긍정 부정 평가

## 사용한 데이터셋
1. DACON 대회 train data, test data
(출처 : https://dacon.io/competitions/official/235864/data)

2. 네이버 리뷰 데이터셋
(출처 : https://raw.githubusercontent.com/e9t/nsmc)

## 목표
0(부정)과 1(긍정)으로 분류된 네이버 영화 리뷰 데이터를 분석하고 딥러닝 모델을 통해 학습시킨 후
새로운 리뷰 데이터를 입력했을 때 리뷰의 긍정과 부정을 분류할 수 있도록 설계한다.

## 사용한 모델
keras 패키지의 라이브러리 사용
1. DNN
2. LSTM(RNN)
