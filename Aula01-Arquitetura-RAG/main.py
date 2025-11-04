import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Defining API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCc1XN7WzYTqY5dtGO4bOgRDn2sk5O8BKs"

# Defining llm
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro-latest", temperature = 0)

print(llm)