import streamlit as st
import re
import string
import joblib
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk

nltk.download('punkt')

# === Load model & vectorizer ===
model = joblib.load("model_sentimen.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# === Inisialisasi Sastrawi ===
stemmer = StemmerFactory().create_stemmer()
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

# === Fungsi Preprocessing Sesuai Notebook ===

def cleaningText(text):
    # Hapus tag HTML
    text = re.sub(r'<.*?>', ' ', text)
    # Hapus angka
    text = re.sub(r'\d+', ' ', text)
    # Hapus karakter spesial dan simbol
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def casefoldingText(text):
    return text.lower()

def fix_slangwords(text):
    # Bisa kamu lengkapi dengan kamus slang sendiri
    slang_dict = {
        'gk': 'tidak',
        'ga': 'tidak',
        'tdk': 'tidak',
        'bgt': 'banget',
        'btw': 'ngomong-ngomong',
        'sm': 'sama'
    }
    return ' '.join([slang_dict[word] if word in slang_dict else word for word in text.split()])

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(tokens):
    return [word for word in tokens if word not in stopwords and len(word) > 1]

def toSentence(tokens):
    return ' '.join(tokens)

# === Streamlit App ===

st.title("Analisis Sentimen Ulasan Aplikasi")
st.write("Masukkan kalimat positif atau negatif untuk dianalisis.")

user_input = st.text_area("Masukkan kalimat:")

if st.button("Analisis"):
    if not user_input.strip():
        st.warning("Teks tidak boleh kosong.")
    else:
        # Preprocessing sesuai pipeline training
        step1 = cleaningText(user_input)
        step2 = casefoldingText(step1)
        step3 = fix_slangwords(step2)
        step4 = tokenizingText(step3)
        step5 = filteringText(step4)
        final_text = toSentence(step5)

        # Transform & predict
        vectorized = vectorizer.transform([final_text])
        prediction = model.predict(vectorized)[0]

        # Output
        if prediction == 1 or prediction == 'positive':
            st.success("✅ Sentimen Positif")
        else:
            st.error("❌ Sentimen Negatif")
