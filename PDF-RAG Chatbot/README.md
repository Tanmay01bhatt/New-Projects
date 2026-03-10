![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

# PDF RAG Chatbot

A **PDF Question Answering system powered by Retrieval-Augmented Generation (RAG)**.  
Users can upload a PDF and interact with it through a conversational interface. The system retrieves relevant document chunks from a vector database and uses **LLMs** to generate context-grounded responses.

The pipeline leverages **LangGraph-based agent workflows**, **semantic retrieval**, and **hallucination detection** to ensure reliable answers based strictly on the uploaded document.

---

## Tech Stack

- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **LLM:** Google Gemini & Groq (LLaMA 3.1)  
- **Embeddings:** HuggingFace (MiniLM)  
- **Vector DB:** FAISS  
- **Framework:** LangChain & LangGraph  
- **DevOps:** Docker & Docker Compose  

---

## Features

- Upload and process PDF documents
- Semantic retrieval over document chunks
- **RAG Fusion retrieval optimization** using Multi-query generation and Reciprocal Rank Fusion (RRF) to improve retrieval accuracy and context relevance
- Hallucination detection to ensure grounded answers
- Out-of-scope question detection
- Conversational chat interface
- FastAPI REST API backend
- Fully containerized deployment

---

## Docker Deployment

Prebuilt Docker images are available on Docker Hub.

### Backend Image
tanmaybhatt18/pdf-ragchatbot-backend:latest

### Frontend Image
tanmaybhatt18/pdf-ragchatbot-frontend:latest

### 1. Pull Docker Images

```bash
docker pull tanmaybhatt18/pdf-ragchatbot-backend:latest
docker pull tanmaybhatt18/pdf-ragchatbot-frontend:latest
```
### 2. Run with Docker Compose

```bash
docker compose up --build
```
### 3. Access the Application
Frontend (Streamlit UI)
```bash
http://localhost:8501
```
Backend API Docs
```bash
http://localhost:8000/docs
```

### Output
![Output](https://raw.githubusercontent.com/Tanmay01bhatt/New-Projects/main/PDF-RAG%20Chatbot/snippet.PNG)
