from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from fastapi import FastAPI

from dotenv import load_dotenv


load_dotenv()

app = FastAPI()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ('system','you are a professional translator'),
    ('user','Detect the language of the following text and translate it to {target_lang}:\n'
     '\n{format_instructions}'
     '\nText : {text}')
])

class ResponseVal(BaseModel):
    detected_language: str = Field(description="Detected source language")
    translated_text: str = Field(description="Translated text")

parser = PydanticOutputParser(pydantic_object=ResponseVal)

chain = prompt | llm | parser

class InputVal(BaseModel):
    text: str
    target_lang: str


@app.post("/translate",response_model=ResponseVal)
def translate_text(req : InputVal):
    result = chain.invoke({
        'text' : req.text,
        "target_lang":req.target_lang,
        "format_instructions": parser.get_format_instructions()
    })
    return result.model_dump()