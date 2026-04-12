# 1. LLMs (Large Language Models)

This directory contains basic demonstrations and implementations of Large Language Models using LangChain.

## 📋 Overview

Large Language Models (LLMs) are powerful AI models trained on vast amounts of text data. This section demonstrates how to interact with and utilize LLMs for various tasks.

## 📁 Files

### `1_llm_demo.py`
Basic LLM demonstration showcasing fundamental capabilities of language models.

**Key Features:**
- Basic LLM initialization
- Simple query-response interactions
- Demonstrates core LLM capabilities

## 🚀 Getting Started

### Prerequisites

```bash
# Ensure you have installed all dependencies
pip install -r ../requirements.txt
```

### Running the Demo

```bash
# From the project root directory
python3 1.LLMS/1_llm_demo.py
```

## 💡 Use Cases

- **Question Answering**: Ask general knowledge questions
- **Text Generation**: Generate creative or informative text
- **Basic Inference**: Understand how LLMs process and respond to prompts

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Understanding LLMs](https://www.langchain.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## 🔧 Configuration

Make sure your `.env` file in the project root contains necessary API keys:

```bash
OPENAI_API_KEY=your_api_key_here
HUGGINGFACE_API_KEY=your_hf_api_key_here
```

## 📝 Example Usage

```python
# Basic LLM query example
from langchain.llms import OpenAI

llm = OpenAI()
response = llm("What is the capital of India?")
print(response)  # Output: The capital of India is New Delhi.
```

## 🐛 Common Issues

- **PyTorch Warning**: Models may not be available without PyTorch, but tokenizers and utilities will still work
- **API Key Errors**: Ensure your API keys are properly configured in the `.env` file

---

**Next Steps**: Explore the `2.ChatModels` directory for more advanced conversational AI implementations.