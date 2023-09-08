import os
from fastapi import FastAPI
import uvicorn
import openai
from dotenv import load_dotenv
from routers import pdf


# Load the environment variables from the .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.include_router(pdf.router)



if __name__== '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
