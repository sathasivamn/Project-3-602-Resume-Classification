import streamlit as st
import pickle
import joblib
import os

import docx
import PyPDF2

# Path to your trained model
MODEL_PATH = "best_model.pkl"

st.set_page_config(page_title="Document Classifier", layout="centered")

# Safe model loader
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except:
        return joblib.load(MODEL_PATH)

model = load_model()

st.title("📄 Resume / Document Classification")
st.write("Upload a document to predict its category")

uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

# Read text from different file types
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)

    elif file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return "\n".join(text)

    return ""

if uploaded_file is not None:
    text = extract_text(uploaded_file)

    if st.button("Predict"):
        if text.strip() == "":
            st.warning("File is empty or unsupported format.")
        else:
            try:
                prediction = model.predict([text])
                st.success(f"✅ Predicted Class: {prediction[0]}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
