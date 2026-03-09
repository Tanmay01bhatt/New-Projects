from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from database import get_retriever
from graph import build_graph

app = FastAPI()

graph = None


class QueryRequest(BaseModel):
    question: str
    thread_id: str


class QueryResponse(BaseModel):
    answer: str



@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):

    global graph

    os.makedirs("temp", exist_ok=True)

    file_path = f"temp/{file.filename}"

    content = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(content)

    retriever = get_retriever(file_path)

    graph = build_graph(retriever)

    return {"status": "PDF processed"}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):

    if graph is None:
        return {"answer": "Please upload a PDF first."}
    
    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }
    result = graph.invoke({
        "question": request.question
    }, config=config)

    return {"answer": result["answer"]}