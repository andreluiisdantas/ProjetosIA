import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.tools.retriever import create_retriever_tool

# Defining API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCc1XN7WzYTqY5dtGO4bOgRDn2sk5O8BKs"

# Defining llm
llm = ChatGoogleGenerativeAI(model = "models/gemini-2.5-flash", temperature = 0)

# Defining file path
file_name = "Código de Conduta e Ética Empresarial da TechCorp S.A.pdf"
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
file_path = os.path.join(base_dir, "Assets", file_name)

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

# Defining embedding models
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Generating the embeddings for a chunks list

## Create a Vector Store (Chroma) of chunks and save
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model
)

# Create Retriever

## Retriever is component, that know how to search in the Vector Store
retriever = db.as_retriever(search_kwargs={"k": 3}) # search_kwargs={"k": 3} It means that it will always look for the three most relevant ones.

## Creating search tool
retriever_tool = create_retriever_tool(
    retriever,
    "busca_codigo_conduta",
    "Busca e retorna informações do Código de Conduta e Ética Empresarial da TechCorp S.A. Use esta ferramenta para responder perguntas sobre o código."
)

## Tools list
tools = [retriever_tool]

## Prompt template, teaching llm think like step-by-step
prompt = hub.pull("hwchase17/react")

# Agent matches llm, tools, prompt
agent = create_react_agent(llm, tools, prompt)

# Execute the agente, (verbose=true is recommended to see the agent's reasoning at the terminal)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Question
question = input("Digite sua pergunta: ")

# Response
response = agent_executor.invoke({"input":  question})

print(f"Pergunta: {question}\n")
print("===================================================")
# A resposta final do agente está na chave 'output'
print(f"Resposta da LLM: {response['output']}\n")
print("===================================================")