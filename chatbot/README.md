# Chatbot Implementations

This directory contains advanced chatbot implementations using LangChain with various prompt templates and message handling techniques.

## 📋 Overview

This section demonstrates how to build production-ready chatbots with proper message handling, conversation history, and prompt templates.

## 📁 Files

### `chatbot.py`
Main chatbot implementation with full conversation capabilities.

**Features:**
- Multi-turn conversations
- Context management
- Response generation
- Integration with various LLM providers

### `chat_prompt_template.py`
Demonstrates ChatPromptTemplate usage in LangChain.

**Features:**
- System message configuration
- User message templates
- Dynamic prompt generation
- Role-based messaging

**Example Use Cases:**
- Customer support agents
- Technical assistants
- Translation services
- Educational tutors

### `message_placeholder.py`
Implements MessagesPlaceholder for dynamic message insertion.

**Features:**
- Dynamic conversation history
- Message placeholder functionality
- Multi-message template handling
- Conversation context preservation

**Key Concepts:**
- Load conversation history from files
- Insert messages dynamically into prompts
- Maintain conversation context
- Handle both user and AI messages

### `messages.py`
Message handling and formatting utilities.

**Features:**
- Message type definitions
- Message serialization/deserialization
- Message formatting utilities
- Conversation history management

### `chat.txt`
Sample conversation history file.

**Purpose:**
- Store conversation history
- Load previous conversations
- Test message placeholder functionality
- Demonstrate conversation persistence

**Format:**
```
HumanMessage(content="User message here")
AIMessage(content="AI response here")
```

## 🚀 Getting Started

### Prerequisites

```bash
# Install dependencies
pip install -r ../requirements.txt

# Ensure you have LangChain core installed
pip install langchain-core
pip install langchain-openai  # or other provider
```

### Configuration

Update your `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
# Add other provider keys as needed
```

### Running Examples

```bash
# Main chatbot
python3 chatbot/chatbot.py

# Chat prompt template example
python3 chatbot/chat_prompt_template.py

# Message placeholder example
python3 chatbot/message_placeholder.py

# Message utilities
python3 chatbot/messages.py
```

## 💡 Key Concepts

### 1. Message Types

#### SystemMessage
Defines the chatbot's role and behavior.
```python
SystemMessage(content="You are a helpful customer support agent")
```

#### HumanMessage
Represents user input.
```python
HumanMessage(content="Where is my order?")
```

#### AIMessage
Represents chatbot responses.
```python
AIMessage(content="Let me check your order status.")
```

### 2. ChatPromptTemplate

Structure conversations with templates:

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}."),
    ("human", "{user_input}")
])
```

### 3. MessagesPlaceholder

Insert dynamic conversation history:

```python
from langchain.prompts import MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])
```

## 📝 Example: Basic Chatbot

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Initialize chat model
chat = ChatOpenAI(temperature=0.7)

# Create prompt template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    ("human", "{user_input}")
])

# Create conversation chain
chain = template | chat

# Get response
response = chain.invoke({"user_input": "Where is my order?"})
print(response.content)
```

## 📝 Example: Conversation with History

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# Initialize
chat = ChatOpenAI()

# Template with history
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# Conversation history
history = [
    HumanMessage(content="What's the weather like?"),
    AIMessage(content="I don't have real-time weather data.")
]

# Create chain
chain = template | chat

# Get response with history
response = chain.invoke({
    "chat_history": history,
    "user_input": "Can you check for New York?"
})
```

## 📝 Example: Loading Chat History from File

```python
import os
from langchain.schema import messages_from_dict, messages_to_dict

# Read chat history
with open("chatbot/chat.txt") as f:
    history_text = f.read()

# Parse messages (custom parsing based on your format)
# Use in your chatbot
```

## 🎯 Chatbot Design Patterns

### 1. Customer Support Bot
```python
template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support agent.
    - Be polite and professional
    - Provide clear solutions
    - Ask for clarification if needed"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])
```

### 2. Technical Assistant
```python
template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert {technology} developer.
    - Provide code examples
    - Explain technical concepts clearly
    - Follow best practices"""),
    ("human", "{question}")
])
```

### 3. Educational Tutor
```python
template = ChatPromptTemplate.from_messages([
    ("system", """You are a patient tutor teaching {subject}.
    - Break down complex topics
    - Use analogies and examples
    - Encourage questions"""),
    MessagesPlaceholder(variable_name="conversation"),
    ("human", "{student_question}")
])
```

## 🔧 Advanced Features

### Memory Management
- Conversation history storage
- Context window management
- Message summarization for long conversations

### Multi-turn Conversations
- Maintain context across turns
- Reference previous messages
- Handle follow-up questions

### Error Handling
- Graceful degradation
- Retry logic
- Fallback responses

## 🐛 Troubleshooting

### Issue: AIMessage Content Not Appearing

**Problem:** Message placeholder not showing AI responses from chat.txt

**Solution:**
1. Verify chat.txt format is correct
2. Ensure proper message parsing
3. Check MessagesPlaceholder variable name matches
4. Validate message objects are properly created

### Issue: File Not Found (chat.txt)

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory: 'chat.txt'`

**Solution:**
```python
import os

# Use absolute path or relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
chat_file = os.path.join(script_dir, "chat.txt")

with open(chat_file) as f:
    history = f.read()
```

### Issue: Context Window Exceeded

**Problem:** Too many messages in conversation history

**Solution:**
- Limit history to recent N messages
- Summarize older conversations
- Use sliding window approach

```python
# Keep only last 10 messages
recent_history = chat_history[-10:]
```

## 📊 Performance Optimization

### 1. Caching
Cache frequent responses to reduce API calls.

### 2. Streaming
Stream responses for better user experience.

```python
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```

### 3. Batch Processing
Process multiple queries efficiently.

## 🔗 Integration Examples

### With Streamlit
```python
import streamlit as st

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:")
if user_input:
    response = chatbot.invoke(user_input)
    st.session_state.messages.append((user_input, response))
```

### With FastAPI
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(message: str):
    response = chatbot.invoke(message)
    return {"response": response}
```

## 📚 Resources

- [LangChain Chat Models](https://python.langchain.com/docs/modules/model_io/chat/)
- [Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)
- [Memory Management](https://python.langchain.com/docs/modules/memory/)
- [Message Types](https://python.langchain.com/docs/modules/model_io/chat/message_types)

## 🔗 Related Directories

- `../Prompt/`: Prompt engineering techniques
- `../2.ChatModels/`: Chat model implementations
- `../3.EmbeddedModels/`: Embeddings for RAG chatbots

---

**Next Steps**: Combine chatbot capabilities with embeddings from `3.EmbeddedModels` to build RAG-powered chatbots.