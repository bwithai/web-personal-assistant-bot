

import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
pinecone.init(api_key="c3642443-3510-43da-b50a-83f0bd5dc845",environment="asia-southeast1-gcp")
os.environ["OPENAI_API_KEY"]="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"

def run_llm(query:str):
   embeddings=OpenAIEmbeddings()
   doc_search=Pinecone.from_existing_index(index_name="assistant-langchain-index",embedding=embeddings)
   chat=ChatOpenAI(verbose=True,temperature=0)
   qa=RetrievalQA.from_chain_type(llm=chat,chain_type="stuff",retriever=doc_search.as_retriever())
   return qa({"query":query})

print(run_llm(query="what is assignemt operator "))


   




