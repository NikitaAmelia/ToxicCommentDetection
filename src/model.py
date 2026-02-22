from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from typing import Tuple
import pandas as pd

def train_svm_model(df: pd.DataFrame) -> Tuple[LinearSVC, TfidfVectorizer]:
    """
    Melatih model SVM dari dataset.
    Kolom wajib: 'tweet' dan 'class'
    """
    # pastikan ada kolom 'clean_tweet'
    if "clean_tweet" not in df.columns:
        df["clean_tweet"] = df["tweet"].astype(str)

    # ubah class jadi binary: (0,1 -> Toxic) dan (2 -> Clean)
    df["class_binary"] = df["class"].replace({0: 0, 1: 0, 2: 1})

    X = df["clean_tweet"]
    y = df["class_binary"]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)

    model = LinearSVC(max_iter=2000)
    model.fit(X_tfidf, y)

    print("[INFO] SVM model trained successfully.")
    return model, vectorizer

def evaluate_model(model, vectorizer, df: pd.DataFrame):
    """
    Mengevaluasi model pada data yang sama.
    """
    X_test = vectorizer.transform(df["clean_tweet"])
    y_test = df["class_binary"]
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
    return report

def predict_label(model, vectorizer, text: str) -> int:
    """
    Prediksi label untuk satu teks.
    """
    vec = vectorizer.transform([text])
    return int(model.predict(vec)[0])