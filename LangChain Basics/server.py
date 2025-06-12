import os
from dotenv import load_dotenv

from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import uvicorn

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
print("=============================================")
print(groq_api_key)
model =ChatGroq(model="gemma2-9b-it", groq_api_key = groq_api_key)

# Create a prompt templete
system_template = "Transulate the following into {language}"
prompt_templete = ChatPromptTemplate.from_messages(
    [
        ('system', system_template),
        ('user', '{text}')
    ]
)

parser = StrOutputParser()
# Create chain
chain = prompt_templete|model|parser

# Application defination
app = FastAPI(title="LangChain Server",
              version='1.0',
              description='A simple API server using LangChain runnable interfaces')

# Add chain routes
add_routes(
    app, chain, path="/chain"
)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port = 8000)

