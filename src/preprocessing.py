import re
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords jika belum
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Membersihkan teks dari RT, URL, mention, angka, tanda baca, dan stopwords.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"\brt\b", "", text)  # hapus "rt"
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # hapus URL
    text = re.sub(r"@\w+|#\w+", "", text)  # hapus mention & hashtag
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # hapus tanda baca
    text = re.sub(r"[^a-z\s]", "", text)  # hanya huruf
    text = re.sub(r"\s+", " ", text).strip()  # rapikan spasi
    
    # hapus stopwords
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text
