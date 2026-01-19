import streamlit as st
import requests
import os

API_URL = "http://backend:8000/recommend"

st.set_page_config(page_title="Anime Recommender", layout="wide")

st.title("Anime Recommendation System")
st.write("Ask for anime recommendations based on your preferences.")

query = st.text_input("Enter your query:")

if st.button("Get Recommendations"):
    if query.strip():
        with st.spinner("Finding the best anime for you..."):
            response = requests.post(
                API_URL,
                json={"question": query}
            )

            if response.status_code == 200:
                answer = response.json()["answer"]
                st.subheader("Recommendations")
                st.write(answer)
            else:
                st.error("Error in Recommendations")
    else:
        st.warning("Please enter a question.")