from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2TokenizerFast

# Initialize the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Create a text splitter instance
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
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