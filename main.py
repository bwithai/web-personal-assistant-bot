import os

from typing import List

from fastapi import FastAPI, UploadFile
from langchain.chat_models import ChatOpenAI
from langchain import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from starlette.middleware.cors import CORSMiddleware

from schemas import AssistantDoResponse
from utils import get_chunks
from dotenv import load_dotenv
load_dotenv()

# app
app = FastAPI(
    title='Personal Assistant for your web',
    version='1.0.0',
    redoc_url='/api/v1/docs'
)

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# global variables
db = None
chat_history = []
chain = None
ConversationalChain = None

api_key = os.environ['OPENAI_API_KEY']


async def train_assistant(chunks):
    global ConversationalChain
    global db
    global chain
    global chat_history

    # Perform the training process
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    chain = load_qa_chain(ChatOpenAI(temperature=0.9), chain_type="stuff")
    ConversationalChain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7, openai_api_key=api_key), db.as_retriever()
    )
    # Clear chat history after training
    chat_history.clear()


@app.post("/api/v1/load-and-train")
async def load_and_split_pdf_files(files: List[UploadFile]) -> dict[str, str]:
    # Create an empty list to store the loaded and split pages
    pages = []

    # Iterate over each uploaded file
    for file in files:
        # Read the uploaded PDF file
        pdf_content = await file.read()

        # Create a temporary file to save the PDF content
        with open(f"uploads/{file.filename}.pdf", "wb") as temp_file:
            temp_file.write(pdf_content)

        # Get the file path of the temporary file
        file_path = os.path.abspath(f"uploads/{file.filename}.pdf")

        # Create a PyPDFLoader instance for the current PDF file
        loader = PyPDFLoader(file_path)

        try:
            # Load and split the pages of the PDF file
            pdf_pages = loader.load_and_split()

            # Extend the 'pages' list with the loaded pages
            pages.extend(pdf_pages)
        except Exception as e:
            return {
                "message": f"Failed to load and split PDF: {str(e)}"
            }
        finally:
            # Delete the temporary file
            os.remove(file_path)

    # Update the global_chunks variable
    chunks = get_chunks(pages)

    # Train Assistant
    await train_assistant(chunks)

    return {
        "message": "Files uploaded and trained successfully",
    }


@app.post("/api/v1/query")
async def request_query(query: AssistantDoResponse):
    global db
    global chain
    global ConversationalChain
    global chat_history

    if db is None:
        return {
            "message": "train your model first",
        }

    actual_text = query.query

    # Concatenate the conversation history with the current input
    full_text = '\n'.join([f"{question}\n{answer}" for question, answer in chat_history])
    full_text += f"\n{actual_text}"

    docs = db.similarity_search(actual_text)
    chain.run(input_documents=docs, question=full_text)

    result = ConversationalChain({"question": full_text, "chat_history": chat_history})
    response_text = result['answer']
    chat_history.append((actual_text, response_text))

    return {
        "History": dict(chat_history),
        "Assistant_Response": result["answer"],
    }