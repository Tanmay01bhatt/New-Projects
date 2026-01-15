import streamlit as  st
import requests
import os

API_URL = os.getenv("API_URL", "http://backend:8000/translate")

st.set_page_config(page_title="Translator",layout="wide")
st.title("Auto Detect Language Translator")

text = st.text_input("Enter text")
target_lang = st.text_input("Target language")

if st.button("Translate"):
    if text and target_lang:
        req = requests.post(API_URL,json={"text":text,"target_lang":target_lang},timeout=30)

        if req.status_code == 200:
            data = req.json()
            st.success(f"Detected Language :{data['detected_language']}")
            st.subheader("Translated Text :")
            st.write(data['translated_text'])
        else:
            st.error("Error in Translation")