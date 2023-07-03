import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone


pinecone.init(
    api_key="c3642443-3510-43da-b50a-83f0bd5dc845",
    environment="asia-southeast1-gcp",
)
os.environ["OPENAI_API_KEY"]="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"
loader=PyPDFLoader("/Users/saeedanwar/Desktop/assitant/documentation-helper/Pine Script v5.pdf")


def ingestion()->None:
    raw_documents=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents[3969:], embeddings, index_name="assistant-langchain-index")
    print("****** Added to Pinecone vectorstore vectors")


if __name__=="__main__":
    ingestion()
