# 2. Chat Models

This directory contains implementations of various chat models using LangChain, including OpenAI, Google, and HuggingFace integrations.

## 📋 Overview

Chat Models are specialized language models designed for conversational interactions. They maintain context and provide more natural dialogue experiences compared to basic LLMs.

## 📁 Files

### `1_chatmodel_openai.py`
OpenAI chat model implementation with translation capabilities.

**Features:**
- OpenAI GPT integration
- English to Hindi translation agent
- Conversational AI capabilities

### `2.chatmodel_google.py`
Google's chat model implementation.

**Features:**
- Google AI integration
- Alternative chat model provider
- Demonstrates multi-provider flexibility

### `hf_api.py`
HuggingFace chat model using API endpoints.

**Features:**
- HuggingFace Inference API integration
- Access to various open-source models
- Cloud-based model execution

**Supported Models:**
- Meta-Llama models
- Mistral models
- Other HuggingFace-hosted models

**Note:** Ensure the model you select is supported by your enabled providers.

### `hf_local.py`
Local HuggingFace model implementation.

**Features:**
- Run models locally without API calls
- Full control over model execution
- Privacy-focused implementation

**Requirements:**
- PyTorch installation required
- Sufficient local compute resources
- Model files downloaded locally

## 🚀 Getting Started

### Prerequisites

```bash
# Install dependencies
pip install -r ../requirements.txt

# For local models, install PyTorch
pip install torch
```

### Configuration

Create or update your `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
```

### Running Examples

```bash
# OpenAI Chat Model
python3 2.ChatModels/1_chatmodel_openai.py

# Google Chat Model
python3 2.ChatModels/2.chatmodel_google.py

# HuggingFace API Model
python3 2.ChatModels/hf_api.py

# HuggingFace Local Model
python3 2.ChatModels/hf_local.py
```

## 💡 Use Cases

- **Translation Services**: English to Hindi and other language pairs
- **Customer Support**: Automated chat support agents
- **Conversational AI**: Interactive dialogue systems
- **Multi-turn Conversations**: Context-aware responses

## 🔧 Troubleshooting

### Common Errors

#### 1. Model Not Supported Error
```
BadRequestError: The requested model 'model-name' is not supported by any provider you have enabled.
```
**Solution:** Use a supported model from your enabled providers or enable additional providers.

#### 2. PyTorch Not Found
```
ImportError: AutoModelForCausalLM requires the PyTorch library
```
**Solution:** Install PyTorch:
```bash
pip install torch
```

#### 3. API Key Issues
```
AuthenticationError: Invalid API key
```
**Solution:** Verify your API keys in the `.env` file.

## 📚 Model Comparison

| Model Type | Pros | Cons | Best For |
|------------|------|------|----------|
| OpenAI | High quality, reliable | Paid service | Production apps |
| Google | Good performance | Limited availability | Google ecosystem |
| HF API | Many model options | Model availability varies | Experimentation |
| HF Local | Privacy, no API costs | Requires compute resources | Offline/private use |

## 📝 Example: Translation Agent

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI()
messages = [
    HumanMessage(content="Translate to Hindi: Hello, how are you?")
]
response = chat(messages)
print(response.content)
```

## 🔗 Related Directories

- `../1.LLMS/`: Basic LLM implementations
- `../chatbot/`: Advanced chatbot implementations with prompt templates
- `../Prompt/`: Prompt engineering examples

---

**Next Steps**: Explore the `3.EmbeddedModels` directory for text embeddings and similarity search.