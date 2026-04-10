# gen-ai

A comprehensive LangChain-based project for exploring Large Language Models (LLMs) and Chat Models with practical implementations.

## 📋 Project Overview

This project demonstrates various implementations of LLMs and Chat Models using LangChain, including:

- Basic LLM demonstrations
- OpenAI Chat Model integrations
- English to Hindi translation agents
- And more...

## 🚀 Features

- **LLM Demos**: Basic implementations showcasing LLM capabilities
- **Chat Models**: OpenAI-based chat model implementations
- **Translation Agent**: English to Hindi translation functionality
- **Modular Structure**: Organized codebase for easy navigation and extension

## 📁 Project Structure

```
gen-ai/
├── 1.LLMS/
│   └── 1_llm_demo.py          # Basic LLM demonstration
├── 2.ChatModels/
│   └── 1_chatmodel_openai.py  # OpenAI chat model with translation agent
├── .gitignore                  # Git ignore configuration
└── README.md                   # Project documentation
```

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd gen-ai
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

- **LangChain**: Framework for developing LLM applications
- **Transformers**: Hugging Face transformers library
- **PyTorch** (optional): For advanced model capabilities
- **OpenAI**: For OpenAI API integration

> **Note**: PyTorch is optional. Without it, models won't be available, but tokenizers, configuration, and file/data utilities will still work.

## 🔧 Configuration

1. **Environment Variables**: Create a `.env` file in the root directory

   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **API Keys**: Ensure you have valid API keys for the services you plan to use

## 💻 Usage

### Running LLM Demo

```bash
python3 1.LLMS/1_llm_demo.py
```

### Running Chat Model with Translation

```bash
python3 2.ChatModels/1_chatmodel_openai.py
```

## 📝 Examples

### Basic LLM Query

The LLM demo can answer questions like:

```
Input: What is the capital of India?
Output: The capital of India is New Delhi.
```

### English to Hindi Translation

The chat model includes a translation agent for English to Hindi conversion.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🐛 Known Issues

- PyTorch dependency warning: Models won't be available without PyTorch installation, but basic functionality remains intact.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Happy Coding! 🚀**
