import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Defining API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCc1XN7WzYTqY5dtGO4bOgRDn2sk5O8BKs"

# Defining llm
llm = ChatGoogleGenerativeAI(model = "models/gemini-2.5-flash", temperature = 0)

# Defining file path
file_name = "Código de Conduta e Ética Empresarial da TechCorp S.A.pdf"
file_path = os.path.join("Assets", file_name)

# Load file
loader = PyPDFLoader(file_path)
documents = loader.load()

# Dividing into chunks

## Text splitter configurantion
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Max limit each chunck characters
    chunk_overlap=200, # Number of overlapping characters between chunks to maintain context.
    length_function=len, # Function to measure text size (default is len)
)

## Applying the split to the uploaded documents
chunks = text_splitter.split_documents(documents)

# Print first chunk
print(chunks[0].page_content)

# Print total chunks
print(f"Número total de chunks: {len(chunks)}")