import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Defining API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCc1XN7WzYTqY5dtGO4bOgRDn2sk5O8BKs"

# Defining llm
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash-latest", temperature = 0)

# Exemple 1: Traditional prompt without RAG

## Main question
question = "What are the work-from-home policies?"

## Prompt template
traditional_prompt = ChatPromptTemplate.from_template(
    "Answer this question: {question}"
)

## Creating chain of execution (Conecting prompt with llm)
traditional_chain = traditional_prompt | llm

## Response from llm
traditional_response = traditional_chain.invoke({"question": question})

print(traditional_response.content)