from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline, Conversation
from pyngrok import ngrok
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import clean_text
from src.model import train_svm_model, predict_label

# === 1. Setup Flask ===
# 🟢 Tambahkan static_folder agar bisa load halle.jpg
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)
CORS(app)

# === 2. Dataset & Model ===
print("[INFO] Loading dataset...")
df = load_data("data/labeled_data.csv")
df["clean_tweet"] = df["tweet"].astype(str).apply(clean_text)
print("[INFO] Training model...")
model, vectorizer = train_svm_model(df)

from transformers import pipeline
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1  # -1 artinya pakai CPU
)

def generate_response(comment, label):
    """Generate balasan positif jika komentar toxic"""
    if label == 0:  # Toxic
        try:
            convo = Conversation(
                f"Someone said something rude: '{comment}'. "
                "Respond kindly and empathetically."
            )
            result = generator(convo)
            reply = result.generated_responses[-1]

            if not reply or len(reply.strip()) < 3:
                reply = "I understand you're upset, but let's stay kind and respectful 🙂"

            print("[DEBUG] Bot reply:", reply)
            return reply

        except Exception as e:
            print("[ERROR] Response generation failed:", e)
            return "Let's try to stay positive and understanding 😊"

    return None  # Kalau bukan toxic, tidak perlu respon


# === 4. Routes ===
@app.route("/", methods=["GET"])
def index():
    """Render halaman utama"""
    return render_template("result.html", comment="", detected_label="", bot_response="")

@app.route("/predict", methods=["POST"])
def predict():
    """Prediksi komentar & hasilkan balasan"""
    if request.is_json:
        data = request.get_json()
        comment = data.get("comment", "")
    else:
        comment = request.form.get("comment", "")

    clean = clean_text(comment)
    label = predict_label(model, vectorizer, clean)
    label_map = {0: "Toxic", 1: "Clean"}
    detected_label = label_map.get(label, "Unknown")
    response = generate_response(comment, label)

    # Jika request dari HTML (form)
    accept = request.headers.get("Accept", "")
    if "text/html" in accept or not request.is_json:
        return render_template(
            "result.html",
            comment=comment,
            detected_label=detected_label,
            bot_response=response if response else "No response needed."
        )

    # Jika API
    return jsonify({
        "comment": comment,
        "detected_label": detected_label,
        "bot_response": response if response else "No response needed."
    })


# === 5. Run Server ===
if __name__ == "__main__":
    port = 5000
    public_url = ngrok.connect(port).public_url
    print("🚀 Public URL:", public_url)
    app.run(port=port)