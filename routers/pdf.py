from fastapi import FastAPI, Request
import os

import traceback
from fastapi import APIRouter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from fastapi import UploadFile, File, Form
import shutil
import tiktoken
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory

load_dotenv()


router = APIRouter()



def tokens_count(string: str) -> int:
      encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
      return len(encoding.encode(string))

@router.post("/trainpdf")
async def trainpdf(pdf_file: UploadFile = File(...), user_id: str = Form(...)):
    if not pdf_file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}

    pdf_folder_path = f"pdf_temp{user_id}"

    if os.path.exists(pdf_folder_path):
        for filename in os.listdir(pdf_folder_path):
            file_path = os.path.join(pdf_folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):  # Check if it's a subdirectory
                    os.rmdir(file_path)  # Delete the subdirectory
            except Exception as e:
                print(f"Error deleting file or directory(Before): {file_path} - {e}")

        # Delete the pdf_folder_path directory itself
        try:
            os.rmdir(pdf_folder_path)
        except Exception as e:
            print(f"Error deleting directory(Before): {pdf_folder_path} - {e}")
    else:
        print(f"PDF folder path not found: {pdf_folder_path}")

    pdf_folder_path = f"pdf_temp{user_id}"



    os.makedirs(pdf_folder_path, exist_ok=True)
    file_path = os.path.join(pdf_folder_path, pdf_file.filename)

    with open(file_path, "wb") as f:
        f.write(await pdf_file.read())

    try:
        documents = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
        doc = []
        for loader in documents:
            doc.extend(loader.load())
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(doc)
        embeddings = OpenAIEmbeddings()
        persist_directory = f'data/chatpdf/{user_id}/{user_id}_pdf_embeddings'
        vectordb = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=persist_directory)
        vectordb.persist()
    except Exception as e:

        for filename in os.listdir(pdf_folder_path):
            file_path = os.path.join(pdf_folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):  # Check if it's a subdirectory
                os.rmdir(file_path)  # Delete the subdirectory
        except Exception as e:
            print(f"Error deleting file or directory: {file_path} - {e}")

    # Delete the pdf_folder_path directory itself
        try:
            os.rmdir(pdf_folder_path)
        except Exception as e:
            print(f"Error deleting directory: {pdf_folder_path} - {e}")

        
        error_message = f"Error Occurred during Data Extraction from Pdf:\n{traceback.format_exc()}"
        with open("error_log.txt", "w") as error_file:
            error_file.write(error_message)
        return {'answer': "Error Occurred during Data Extraction from Pdf. Please check 'error_log.txt' for more details."}
    


    for filename in os.listdir(pdf_folder_path):
        file_path = os.path.join(pdf_folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):  # Check if it's a subdirectory
                os.rmdir(file_path)  # Delete the subdirectory
        except Exception as e:
            print(f"Error deleting file or directory: {file_path} - {e}")

    # Delete the pdf_folder_path directory itself
    try:
        os.rmdir(pdf_folder_path)
    except Exception as e:
        print(f"Error deleting directory: {pdf_folder_path} - {e}")

  

    return {'answer': "PDF EMBEDDINGS GENERATED SUCESSFULLY"}

   
    # return {"message": "File Embedding successfully."}

@router.post('/chatpdf')
async def chatpdf(request: Request, data: dict):
    
    user_id = data.get('user_id')
    query = data.get('query')
    key = data.get('key','1')
    # Run the search code and get the results
    embeddings = OpenAIEmbeddings()

    if key == '1' :
        persist_directory = f'data/chatpdf/{user_id}/{user_id}_pdf_embeddings'
        prompt_template = """You are a chatbot having a conversation with a human using 
                        the parts of a long document and a question.(Do not use your own knowledge to answer the question limit yourself to the long document 
                        provided to you to answer the questions)
                        Your task is to use the long document for context and create a final answer and never answer a question if there is no
                        context related to it.

                        Always follow the following Instructions:
                        - Carefully read the provided 'Context' to understand the document's information.
                        - Use the information in 'Context' to answer the 'Question.'
                        - If there is no relevant information in the 'Context' to answer the 'Question',
                          reply appropriately without providing a specific answer.
                        {context}
                        {chat_history}
                        Human: {human_input}
                        Chatbot:"""
    elif key=='2':
        persist_directory = f'data/islamic_data/collection'
        prompt_template =  """Your name is Habibi AI you are a chatbot having a conversation with a human only about different Islam related topics Answer the humans questions using 
                        the parts of a long document only.Do not answer the humans question if the question is not related to the long document .
                        Use the long document to generate a detailed answer to the humans question.
                        Always follow the following Instructions:
                        - Carefully read the provided 'long document' to understand the document's information.
                        - Use the information in 'long document' to answer the 'Question.'
                        - If there is no relevant information in the 'lomg document' to answer the 'Question',
                          reply appropriately without providing a specific answer. 
                        -Only answer questions related to Islamic topics 
                        -Never answer questions that are unrelated to the long document 
                        {context}
                        {chat_history}
                        Human: {human_input}
                        Chatbot:"""
        
    elif key=='3':
        persist_directory = f'data/financial_data/collection'
        prompt_template = """Your name is Morfinity AI you are a chatbot having a conversation with a human about financial topics use
                    the parts of a long document as context to answer the humans questions.Use the long document as context and give a detailed answer to the 
                    humans questions

                    Always follow the following Instructions:
                        - Carefully read the provided 'Context' to understand the document's information.
                        - Use the information in 'Context' to help answer the 'Question.'
                        - If there is no relevant information in the 'Context' to answer the 'Question',
                          reply appropriately.
                        -Only answer questions related to finance or other similar topics .If the question is not related to these topics respond 
                        appropriately without providing a specific answer. 


                    {context}
                    {chat_history}
                    Human: {human_input}
                    Chatbot:"""

    else:
           return {'Message': "Incorrect key"}


    PROMPT = PromptTemplate(input_variables=["chat_history", "human_input", "context"], template=prompt_template)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k',max_tokens=1000)
    chain = load_qa_chain(llm=llm, memory=memory, chain_type="stuff", prompt=PROMPT)

    Vectordb = Chroma(persist_directory=persist_directory, embedding_function= embeddings)
    retriever = Vectordb.as_retriever(search_type="mmr",kwargs=8)
    docs = retriever.get_relevant_documents(query)
    ans = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
    result = ans['output_text']
   
    return {'answer': result}


@router.post('/deletepdf')
async def deletepdf(request: Request, data: dict):
    user_id = data.get('user_id')
    delete_directory =f'data/chatpdf/{user_id}'

    # Check if the directory exists
    if os.path.exists(delete_directory):
        # Delete the directory and its contents
        shutil.rmtree(delete_directory)
        return {'answer': f"Pdf Embeddings for user {user_id} deleted successfully."}
     
    else:
        return {'answer': f"No Pdf embeddings found for user {user_id}."}
    


@router.post('/reset')
async def deletepdf(request: Request, data: dict):
    reset = data.get('reset', 'False')

    if reset == 'True' or reset== 'true':
        delete_directory = 'data'

        # Check if the directory exists
        # Check if the directory exists
        if os.path.exists(delete_directory):
            try:
                # Delete the directory and its contents
                shutil.rmtree(delete_directory)
                return {'answer': "Reset Successful"}
            except Exception as e:
                return {'answer': f"Failed to reset: {str(e)}"}

        else:
            return {'answer': "No Data exists"}

    else:
        return {'answer': "If reset is True, only then all data will be deleted"}
    





