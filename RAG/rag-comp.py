from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.chat_message_histories import ChatMessageHistory #Stores messages
from langchain_core.runnables.history import RunnableWithMessageHistory#Feeds messages into the chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever#Handles question rewriting based on chat history
import uuid

load_dotenv()


loader = PyPDFLoader("Tree of Thoughts.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)


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
    ("system", "You are a helpful assistant. Answer ONLY using the provided context."),
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


store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


session_id = str(uuid.uuid4())
print("\nConversational RAG Bot (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    response = chat_rag_chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": session_id}}
    )

    print("Bot:", response)