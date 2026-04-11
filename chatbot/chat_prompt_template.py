from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

chat_template = ChatPromptTemplate([
    ("system","You a a helpful AI Asstent having knowledge of {domain}."),
    ("human","Explain in simple term, what is {topic}")
])

prompt = chat_template.invoke({"domain":"cricket", "topic":"Helicoptor"})

print(prompt)