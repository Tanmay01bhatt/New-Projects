from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory #Stores messages
from langchain_core.runnables.history import RunnableWithMessageHistory#Feeds messages into the chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever#Handles question rewriting based on chat history
import uuid

load_dotenv()

def data_ingestion(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    return chunks

def vector_embedding(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question, "
     "rewrite the question so it is a standalone question."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}")   
    ])
    history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    q_prompt
    )
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer using the given context."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "Context:\n{context}\nQuestion:\n{question}")
    ])

    rag_chain = (
    {
        "context": RunnableLambda(
            lambda x: history_aware_retriever.invoke({
                "input": x["question"],
                "chat_history": x.get("chat_history", [])
            })
        ),
        "question": RunnableLambda(lambda x: x["question"]),
    }
    | prompt
    | llm
    | StrOutputParser()
    )
    return rag_chain


store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#temp dir to store uploaded file
import os

def get_file_path(file_upload):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file_upload.name)
    with open(file_path, "wb") as f:
        f.write(file_upload.getbuffer())

    return file_path    

# app
st.set_page_config(page_title="RAG Chat Bot")
st.title("RAG Chatbot with Chat History")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_ui" not in st.session_state:
    st.session_state.chat_ui = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

uploaded_file = st.file_uploader( "Upload a PDF",type=["pdf"],key="pdf_uploader")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        file_path = get_file_path(uploaded_file)
        chunks = data_ingestion(file_path)
        st.session_state.vectorstore = vector_embedding(chunks)
        st.success("PDF processed and vector store created.")
        rag_chain = rag_chain(st.session_state.vectorstore)

        st.session_state.chat_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history")
        
        

    st.success("PDF processed successfully!")


for role, msg in st.session_state.chat_ui:
    with st.chat_message("user" if role == "You" else "assistant"):
        st.markdown(msg)

query = st.chat_input("Ask a question about the PDF")

with st.spinner("Generating response..."):
    if query:
    # Save user message
        st.session_state.chat_ui.append(("You", query))

        response = st.session_state.chat_rag_chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": st.session_state.session_id}}
        )

        # Save bot message
        st.session_state.chat_ui.append(("Bot", response))

    #  Force rerun so full history re-renders(show chat history)
        st.rerun()