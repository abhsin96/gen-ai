from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages = [
    SystemMessage("You are a AI assistant"),
    HumanMessage("Tell me about sky under 200 characters")
]

result = model.invoke(messages)

messages.append(AIMessage(result.content))

print(messages)