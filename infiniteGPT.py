"""
    Run our text file through infiniteGPT to clean its grammar and punctuation.
"""

import os
import openai
from langchain.chat_models.openai import ChatOpenAI
import tiktoken
from pathlib import Path
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=500, openai_api_key=api_key)



def load_text(file_path):
    with Path(file_path).open("r") as file:
        return file.read()


def save_to_file(responses, output_file):
    with Path(output_file).open('w') as file:
        file.write("\n".join(responses))


def call_openai_api(chunk):
    messages = [
        SystemMessage(
            content="Clean the following transcripts of all gramatical mistakes, misplaced words, and identify the speakers."),
        HumanMessage(content=chunk)
    ]
    response = chat(messages)
    return response.content.strip()


def split_into_chunks(text, n_tokens=300):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), n_tokens):
        chunks.append(' '.join(encoding.decode(tokens[i:i + n_tokens])))
    return chunks


def process_chunks(input_file, output_file, delay=0):  # delay in seconds (if you hit a rate limit error)
    text = load_text(input_file)
    chunks = split_into_chunks(text)[:5]
    responses = []
    for chunk in tqdm(chunks):
        responses.append(call_openai_api(chunk))

    save_to_file(responses, output_file)

    # Specify your input and output files
