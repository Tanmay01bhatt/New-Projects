import streamlit as st
import requests
import uuid
import os

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="PDF RAG", layout="wide")

st.title("PDF RAG ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and not st.session_state.pdf_uploaded:

     with st.spinner("Processing PDF..."):

        files = {"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")}

        response = requests.post(
            f"{API_URL}/upload_pdf",
            files=files
        )

        if response.status_code == 200:
            st.session_state.pdf_uploaded = True
            st.success("PDF processed successfully")
        else:
            st.error(f"Error processing PDF:{response.text}")

#show chat history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ask question
prompt = st.chat_input("Ask a question about the PDF")

if prompt:

    # show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": prompt,
            "thread_id": st.session_state.thread_id
        }
    )

    if response.status_code == 200:

        answer = response.json()["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    else:
        st.error("Error getting answer")
        