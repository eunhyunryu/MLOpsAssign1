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

# HC3 dataset
data = hgds['train']

# human answers
human_rows = []
for row in data:
    for ans in row['human_answers']:
        human_rows.append({'text': ans, 'label': 0})  # 0: human

# ChatGPT answers
ai_rows = []
for row in data:
    for ans in row['chatgpt_answers']:
        ai_rows.append({'text': ans, 'label': 1})  # 1: AI

# pandas dataframe transformation
df_human = pd.DataFrame(human_rows)
df_ai = pd.DataFrame(ai_rows)

# concat
df = pd.concat([df_human, df_ai]).sample(frac=1).reset_index(drop=True)

nltk.download('stopwords')
nltk.download('wordnet')

def text_to_words(document):
    stop_words = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    # lowering words and removing stopwords
    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stop_words])
    # removing punctuation
    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)
    # lemmatize
    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())

    return normalized

# new cleaned text column
df["clean_text"] = df["text"].apply(text_to_words)

# add word count feature
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

# add text length feature
df['text_length'] = df['clean_text'].apply(len)

# add average word length feature
df['avg_word_length'] = df['clean_text'].apply(lambda x: sum(len(w) for w in x.split()) / (len(x.split())+1e-5))

# add number of exclamations feature
df['num_exclamations'] = df['text'].apply(lambda x: x.count('!'))

# add number of questions feature
df['num_questions'] = df['text'].apply(lambda x: x.count('?'))

# add sentiment analysis feature
# pos, neg, neutral
df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
# objective, subjective
df['subjectivity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# check class imbalance
print(df['label'].value_counts())


# 1. random forest without full text

start_time1 = time.time()

X = df.drop(columns=["text", "label", "clean_text"])

y = df["label"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE 
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)

grid_search.fit(X_resampled, y_resampled)

# best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

end_time1 = time.time()

# Confusion Matrix visualization
ConfusionMatrixDisplay.from_estimator(best_model, X_test_scaled, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance visualization
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

# random forest with full text
start_time2 = time.time()

X = df["clean_text"]
y = df["label"]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)  
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, stratify=y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

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

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

end_time2 = time.time()

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("Elapsed Time :", end_time2 - start_time2)

