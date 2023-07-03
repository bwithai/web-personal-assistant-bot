import streamlit as st
from streamlit_chat import message


import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import pinecone

pinecone.init(api_key="c3642443-3510-43da-b50a-83f0bd5dc845",environment="asia-southeast1-gcp")
os.environ["OPENAI_API_KEY"]="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"

def load_chain(query:str):
   embeddings=OpenAIEmbeddings()
   doc_search=Pinecone.from_existing_index(index_name="assistant-langchain-index",embedding=embeddings)
   memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
   qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), doc_search.as_retriever(), memory=memory)
   chain=qa({"question": query})
   return chain



# print(run_llm(query="what is assignment operator "))


# chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = load_chain(query=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")






