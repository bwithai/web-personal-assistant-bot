import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2TokenizerFast
from pathlib import Path

# Initialize the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Create a text splitter instance
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=lambda x: len(tokenizer.encode(x))
)


def get_chunks(pages):
    # Write pages to a file
    with open("uploads/data.txt", "w") as f:
        f.write('\n'.join(str(page) for page in pages))

    # Read the contents from the file
    text = open("uploads/data.txt", "r").read()

    # Perform the chunking process
    chunks = text_splitter.create_documents([text])

    # Calculate token counts for each chunk
    token_counts = [len(tokenizer.encode(chunk.page_content)) for chunk in chunks]

    print("Total tokens required: ", token_counts)

    return chunks


def remove_duplicates(clean_data):
    """
        We start by constructing our knowledge base.
    """

    # Read the .txt file into Python
    with Path(clean_data).open('r') as file:
        lines = file.read().splitlines()

    # Group the lines into chunks of 5
    chunks = [' '.join(lines[i:i + 5]) for i in range(0, len(lines), 5)]

    # Convert list of chunks into a DataFrame
    data = pd.DataFrame(chunks, columns=['context'])

    # Add an index column and a name column
    data['name'] = 'PDF'

    # Remove duplicates (if any)
    data.drop_duplicates(subset='context', keep='first', inplace=True)

    print(data)
