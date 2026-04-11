from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

# chat template

chat_template = ChatPromptTemplate([
    ("system","You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{query}")
])

chat_history = []
# load chat history

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
chat_file_path = os.path.join(script_dir, "chat.txt")

with open(chat_file_path) as f:
    chat_template.append(f.readline())

prompt = chat_template.invoke({"chat_history":chat_history, "query":"Where is my order"})

print(prompt)