from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from pydantic import BaseModel
from fastapi import FastAPI


from dotenv import load_dotenv


load_dotenv()

app = FastAPI()

loader = CSVLoader(file_path="anime.csv" ,encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(chunks, embeddings)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an anime recommendation expert that recommends anime from the provided context and user preferences."),
    ("human", "User Query : {question}"
     "\n\nContext : {context}"
     "\n\nRecommend 3 anime based on the context and give a brief explanation for each recommendation.")
])

def format_docs(docs): # combines multiple docs into 1 string
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {'context':retriever | format_docs,
     'question': RunnablePassthrough()}
     |prompt
     |llm
     |StrOutputParser()

)
# FAst API

app = FastAPI()

class InputVal(BaseModel):
    question: str

class ResponseVal(BaseModel):
    answer: str

@app.post("/recommend", response_model=ResponseVal)
def recommend_anime(request: InputVal):  
    response = rag_chain.invoke(request.question)
    return {"answer": response}