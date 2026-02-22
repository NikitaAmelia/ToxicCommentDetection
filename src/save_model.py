# src/save_model.py
import joblib
from src.data_loader import load_data
from src.preprocessing import clean_text
from src.model import train_svm_model

print("🚀 Training model...")

# 1. Load dataset
df = load_data("data/labeled_data.csv")
df["clean_tweet"] = df["tweet"].astype(str).apply(clean_text)

# 2. Train model
model, vectorizer = train_svm_model(df)

# 3. Save model & vectorizer ke root folder
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model & vectorizer saved as model.pkl & vectorizer.pkl")
