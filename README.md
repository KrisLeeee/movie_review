# movie_review
Movie Reviews Sentiment Analysis (Positive vs Negative assessment)

영화 리뷰의 긍정 부정 평가


## Objective (목표)

This project analyzes Naver movie review data(Korean) labeled with binary sentiment (0 for negative, 1 for positive), and trains a deep learning model to classify the sentiment. The model is capable of predicting whether a newly given review expresses a positive or negative opinion.

0(부정)과 1(긍정)으로 분류된 네이버 영화 리뷰 데이터(한글)을 분석하고 딥러닝 모델을 통해 학습시킨 후 새로운 리뷰 데이터를 입력했을 때 리뷰의 긍정과 부정을 분류할 수 있도록 설계한다.


## Used Dataset (사용한 데이터셋)
1. Movie Review Dataset from DACON (Korean AI Hackaton Platform) AI 해커톤 플랫폼 DACON [Source of Data](https://dacon.io/competitions/official/235864/data)
  - 5000 train data, 5000 test data

2. Movie Review Dataset gathered from Naver (Korean Web Portal) 네이버 영화 리뷰 데이터셋 [Source of Data](https://raw.githubusercontent.com/e9t/nsmc)
  - 150000 train data, 50000 test data -> Sampled in 10000 train, 5000 test due to google colab RAM memory shortage


## Workflow 과정

1. Data Analysis 데이터 분석

  - Analyze overall review : review length distribution, number of used words, often used words
  - Compare positive and negative reviews : number of reviews, review length distribution, number of used words, often used words

![image](https://github.com/user-attachments/assets/e75f2c61-d1b6-421d-a795-2c47019bd0db)
Review Length
![image](https://github.com/user-attachments/assets/81fb1bdc-31bf-4276-b939-f97cdfe6d4e4)
Word Counts


  - Examine the top 10 frequent words that are used in positive reviews, negative reviews, and both

<img src="https://github.com/user-attachments/assets/8e08d96b-a814-489a-83be-e7a1edd3bb4d" width="400">
<img src="https://github.com/user-attachments/assets/b946bbc7-6c61-4534-9713-c0d3e35918ae" width="500">

> {'영화': 1501, '정말': 320, '진짜': 269, '최고': 220, '평점': 185, '연기': 181, '감동': 161, '재미': 152, '보고': 140, '스토리': 130}

> 공통으로 등장하는 단어들 Top10 :  ['영화', '진짜', '평점', '재미', '정말', '스토리', '시간', '내용', '감독', '그냥']

<img src="https://github.com/user-attachments/assets/1f63d321-d236-4c7d-b662-2009c85db656" width="400">
<img src="https://github.com/user-attachments/assets/42f5420d-7c52-4933-956a-b19dbfa5abee" width="500">

> 긍정 리뷰 Top10: [('최고', 214), ('사랑', 72), ('다시', 70), ('명작', 60), ('가슴', 51), ('역시', 44), ('인생', 41), ('지금', 39), ('매력', 35), ('마음', 34)]

<img src="https://github.com/user-attachments/assets/de64726f-c12f-4093-9f0b-15d88c422d5e" width="400">
<img src="https://github.com/user-attachments/assets/80da43b4-766c-403a-b99e-7c8b88d6fcdf" width="500">

> 부정 리뷰 Top10: [('최악', 98), ('쓰레기', 70), ('이건', 55), ('별로', 53), ('실망', 36), ('수준', 36), ('점도', 34), ('원작', 31), ('무슨', 31), ('코미디', 29)]


2. Tokenization 토큰화

  - Tokenization is the process of splitting text into smaller units such as words or subwords, which are used as the basic input for NLP models.
    
  - Use Okt from konlpy library
  - Remove irrelevant characters and extra spaces:
    ```
    data['preprocessed'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 한글과 공백을 제외하고 모두 제거
    data['preprocessed'] = data['preprocessed'].str.replace(" +", " ") # 다중 공백 제거
    ```
  - Extract and output only the morphemes that are relevant to distinguishing between positive and negative sentiment : Nouns, Verbs, Adverbs, Adjectives

    `main_words = [word_pos[0] for word_pos in pos if word_pos[1] in ("Noun", "Adverb", "Adjective", "Verb")]`

    ![image](https://github.com/user-attachments/assets/8136a0de-320a-4bac-9d3f-387bafcc79ee)


3. Experimented Model Structure 모델 실험

  - Countervectorizer : CountVectorizer extracts word frequency features from the training data and converts them into numerical vectors.
    ![image](https://github.com/user-attachments/assets/ab12a32e-752d-49f1-b921-e91a2ebc63ac)

    ```
    vectorizer = CountVectorizer() #countvectorizer 생성
    vectorizer.fit(X_train) # countvectorizer 학습
    X_train_vec = vectorizer.transform(X_train) # transform
    ```
  - Logistic Regression (skit learn library)
      - Logistic Regression accuracy = 0.7752
    ```
    from sklearn.linear_model import LogisticRegression #모델 불러오기
    model1 = LogisticRegression() #객체에 모델 할당
    model1.fit(X_train_vec, y_train) #모델 학습
    ```
    
  - Soft Vector Machine (skit learn library)
      - Soft Vector Machine accuracy = 0.7556
    ```
    from sklearn import svm
    model2 = svm.SVC()  #객체에 모델 할당
    model2.fit(X_train_vec, y_train) #모델 학습
    ```
  - DNN (tenserflow keras) : 3층 신경망 + Dropout(0.4)
      - loss: 0.2614 - acc: 0.8960 - val_loss: 0.4337 - val_acc: 0.7860
      - 3 Layer Deep Neural Network with Dropout
      - Hyperparameter :
        > Activation = 'sigmoid', Dropout = 0.4, Optimizer = 'rmsprop', Loss Function = 'sparse_catagorial_crossentropy', Epoch = 40, Batch Size = 150
        <img src="https://github.com/user-attachments/assets/25e6339a-1fce-44bf-b3e0-612c342ef429" width="300">
        <img src="https://github.com/user-attachments/assets/87b7724a-7130-40e4-a646-a61b23ac7a3f" width="300">


  - RNN by LSTM (tensorflow keras)
    - loss: 0.6932 - acc: 0.5171 - val_loss: 0.6952 - val_acc: 0.4940
    - 2 Layer RNN Network with 100 Embedding Dimension and 128 Hidden Units
    - Hyperparameter :
      > Activation = 'sigmoid', Optimizer = 'rmsprop', Loss Function = 'binary_crossentropy', Epoch = early stops at 11
      <img src="https://github.com/user-attachments/assets/7425edd0-0b64-44a0-8d75-7327e8178176" width="300">
      <img src="https://github.com/user-attachments/assets/51197df9-a724-4dc9-acc0-b1318b1b470c" width="300">


    - As the sequence metters, instead of countervectorizer, "keras Tokenizer" is used.
      - After splitting every words, assigns every word with a index number
      - Tokenized train data : Each sentence is in a list, containing words in index number.
        ```
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        tokenizer = Tokenizer() # 토큰화
        tokenizer.fit_on_texts(word_split) # 단어 인덱스 구축    - Tokenizer : 단어의 빈도수와  추출
        ```
        <img src="https://github.com/user-attachments/assets/7f441504-a222-4e2f-b278-4a46590a7d8f" width="400">
        <img src="https://github.com/user-attachments/assets/1a474ea2-4927-4624-9836-c382d2e9ffb8" width="400">

      - As sentences have different length, padding to match the longest list is required
        ![image](https://github.com/user-attachments/assets/8b846110-b8d7-4d91-9ff7-f0eec514d542)

  - Additional Trials
      - CNN
      - Multiple models mixed

## Test Evaluation 

1. RNN by LSTM
   
   ![image](https://github.com/user-attachments/assets/0cec5dbf-82a3-4af9-85c3-5d090cae39ab)
   > Red : Possibility of Positive Review / Blue : Possibility of Negative Review
   
   Many predictions cluster around 0.5 for both classes, indicating uncertainty and contributing to the model’s lower performance.


3. DNN : Best Performance Observed
   
   ![image](https://github.com/user-attachments/assets/ed864a96-7cc2-4e65-bf4f-ffa34c1e0709)
   > Red : Possibility of Positive Review / Blue : Possibility of Negative Review

    Predictions are clearly biased toward 0 or 1, showing confident classifications.
    Compared to the RNN, the number of ambiguous (near 0.5) cases is significantly lower, leading to better performance.
