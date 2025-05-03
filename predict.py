from datasets import load_dataset
import pandas as pd

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

import matplotlib.pyplot as plt
import seaborn as sns

import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score
)

hgds = load_dataset("Hello-SimpleAI/HC3", "all")

# HC3 데이터셋에서 train split 불러오기
data = hgds['train']

# 사람 답변
human_rows = []
for row in data:
    for ans in row['human_answers']:
        human_rows.append({'text': ans, 'label': 0})  # 0: 사람 말투

# 챗GPT 답변
ai_rows = []
for row in data:
    for ans in row['chatgpt_answers']:
        ai_rows.append({'text': ans, 'label': 1})  # 1: AI 말투

# 판다스 데이터프레임으로 변환
df_human = pd.DataFrame(human_rows)
df_ai = pd.DataFrame(ai_rows)

# 합치기
df = pd.concat([df_human, df_ai]).sample(frac=1).reset_index(drop=True)

nltk.download('stopwords')
nltk.download('wordnet')

def text_to_words(document):
    stop_words = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    # 소문자화 및 불용어 제거
    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stop_words])
    # 구두점 제거
    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)
    # 표제어 추출
    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())

    return normalized

# 기존 text 컬럼에 함수 적용해서 새 컬럼 생성
df["clean_text"] = df["text"].apply(text_to_words)

# 단어수 feature 추가
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

# 글자수 feature 추가
df['text_length'] = df['clean_text'].apply(len)

# 사용한 단어의 길이 평균 feature 추가
df['avg_word_length'] = df['clean_text'].apply(lambda x: sum(len(w) for w in x.split()) / (len(x.split())+1e-5))

# 느낌표 사용 횟수
df['num_exclamations'] = df['text'].apply(lambda x: x.count('!'))

# 물음표 사용 횟수
df['num_questions'] = df['text'].apply(lambda x: x.count('?'))

# 감성분석 feature 추가
# 긍정, 부정, 중립
df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
# 주관적, 객관적
df['subjectivity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# 레이블 불균형 확인
print(df['label'].value_counts())


# 1. 텍스트를 활용하지 않은 랜덤포레스트 모델

start_time1 = time.time()
# X: text와 label을 제외한 모든 컬럼
X = df.drop(columns=["text", "label", "clean_text"])

# y: label 컬럼만
y = df["label"]

# 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE 적용
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# 하이퍼파라미터 튜닝 (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)

grid_search.fit(X_resampled, y_resampled)

# 최적 모델
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

end_time1 = time.time()

# Confusion Matrix 시각화
ConfusionMatrixDisplay.from_estimator(best_model, X_test_scaled, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance 시각화
importances = best_model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title('Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("Elapsed Time :", end_time1 - start_time1)

# 2. 텍스트 데이터를 활용한 랜덤 포레스트 모델
start_time2 = time.time()

X = df["clean_text"]
y = df["label"]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=1000)  # 필요에 따라 max_features 조정
X_vectorized = vectorizer.fit_transform(X)

# 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, stratify=y, test_size=0.2, random_state=42
)

# SMOTE 적용
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 하이퍼파라미터 튜닝 (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_search.fit(X_resampled, y_resampled)

# 최적 모델
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

end_time2 = time.time()

# Confusion Matrix 시각화
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# 성능 지표 출력
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("Elapsed Time :", end_time2 - start_time2)

