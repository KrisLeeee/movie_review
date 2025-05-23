# movie_review
Movie Reviews Sentiment Analysis (Positive vs Negative assessment)

영화 리뷰의 긍정 부정 평가


## Used Dataset (사용한 데이터셋)
1. Movie Review Dataset from DACON (Korean AI Hackaton Platform) AI 해커톤 플랫폼 DACON [Source of Data](https://dacon.io/competitions/official/235864/data)
  - 5000 train data, 5000 test data


2. Movie Review Dataset gathered from Naver (Korean Web Portal) 네이버 영화 리뷰 데이터셋 [Source of Data](https://raw.githubusercontent.com/e9t/nsmc)
  - 150000 train data, 50000 test data -> Sampled in 10000 train, 5000 test due to google colab RAM memory shortage


## Objective (목표)

This project analyzes Naver movie review data labeled with binary sentiment (0 for negative, 1 for positive), and trains a deep learning model to classify the sentiment. The model is capable of predicting whether a newly given review expresses a positive or negative opinion.

0(부정)과 1(긍정)으로 분류된 네이버 영화 리뷰 데이터를 분석하고 딥러닝 모델을 통해 학습시킨 후 새로운 리뷰 데이터를 입력했을 때 리뷰의 긍정과 부정을 분류할 수 있도록 설계한다.


## Model Structure (모델 설계)
TensorFlow keras Library used.


1. DNN
2. LSTM(RNN)


## Workflow 과정

1. Data Analysis 데이터 분석
  - 긍정과 부정 리뷰 각각의 개수, 리뷰의 길이, 단어 수 비교
  - 빈도 수가 높은 단어 상위 10개

2. Tokenize 토큰화
  - konlpy의 Okt 이용
  - 명사, 부사, 형용사,동사 : 긍정과 부정의 판별에 의미가 있는 형태소만 출력

3. Build Model 모델 설계
  - Logistic Regression (skit learn library)
  - Soft Vector Machine (skit learn library)
  - DNN (tenserflow keras) : 3층 신경망 + Dropout(0.4)
    - Countervectorizer : 단어의 빈도수만 추출
  - RNN by LSTM (tensorflow keras)
    - Tokenizer : 단어의 빈도수와  추출
