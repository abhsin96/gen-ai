# Prompt Engineering

This directory contains examples and utilities for prompt engineering and template generation using LangChain.

## 📋 Overview

Prompt engineering is the art and science of crafting effective prompts to get the best results from language models. This directory demonstrates various prompt techniques and UI tools.

## 📁 Files

### `generate_prompt.py`
Prompt generation utilities and examples.

**Features:**
- Dynamic prompt creation
- Template-based prompt generation
- Variable substitution
- Prompt optimization techniques

### `prompt_ui.py`
Streamlit-based UI for interactive prompt engineering.

**Features:**
- Interactive web interface
- Real-time prompt testing
- Template management
- Visual prompt builder
- Response visualization

## 🚀 Getting Started

### Prerequisites

```bash
# Install dependencies
pip install -r ../requirements.txt

# Ensure Streamlit is installed
pip install streamlit
```

### Running the Prompt UI

```bash
# Launch the Streamlit app
streamlit run Prompt/prompt_ui.py
```

The UI will open in your default browser at `http://localhost:8501`

### Running Prompt Generation

```bash
python3 Prompt/generate_prompt.py
```

## 💡 Prompt Engineering Techniques

### 1. Zero-Shot Prompting
Direct instruction without examples.

```python
prompt = "Translate the following English text to Hindi: {text}"
```

### 2. Few-Shot Prompting
Provide examples to guide the model.

```python
prompt = """
Translate English to Hindi:

English: Hello
Hindi: नमस्ते

English: Thank you
Hindi: धन्यवाद

English: {text}
Hindi:
"""
```

### 3. Chain-of-Thought
Encourage step-by-step reasoning.

```python
prompt = """
Solve this problem step by step:

Problem: {problem}

Let's think through this:
1.
"""
```

### 4. Role-Based Prompting
Assign a specific role to the model.

```python
prompt = """
You are an expert Python developer.
Help me with: {question}
"""
```

## 🎨 Prompt UI Features

### Template Management
- Save and load prompt templates
- Organize templates by category
- Version control for prompts

### Interactive Testing
- Test prompts in real-time
- Compare different prompt variations
- View response quality metrics

### Variable Management
- Define dynamic variables
- Preview prompt with sample data
- Validate variable substitution

## 📝 Example: Using Prompt Templates

```python
from langchain.prompts import PromptTemplate

# Create a template
template = PromptTemplate(
    input_variables=["topic", "audience"],
    template="""
    Write a {audience}-friendly explanation of {topic}.
    
    Make it:
    - Clear and concise
    - Easy to understand
    - Engaging and informative
    """
)

# Use the template
prompt = template.format(
    topic="machine learning",
    audience="beginner"
)
```

## 📝 Example: Chat Prompt Template

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}."),
    ("human", "{user_input}")
])

prompt = template.format_messages(
    role="customer support agent",
    user_input="Where is my order?"
)
```

## 🔧 Best Practices

### 1. Be Specific
❌ Bad: "Tell me about AI"
✅ Good: "Explain the difference between supervised and unsupervised learning in 3 sentences"

### 2. Provide Context
❌ Bad: "Translate this: {text}"
✅ Good: "Translate the following customer support message from English to Hindi: {text}"

### 3. Set Constraints
❌ Bad: "Write a story"
✅ Good: "Write a 200-word story about a robot, suitable for children aged 8-10"

### 4. Use Examples
❌ Bad: "Format this data"
✅ Good: "Format this data like this example: [show example]"

### 5. Iterate and Test
- Test prompts with various inputs
- Measure response quality
- Refine based on results

## 🎯 Prompt Optimization Checklist

- [ ] Clear and specific instructions
- [ ] Appropriate context provided
- [ ] Constraints and requirements defined
- [ ] Examples included (if needed)
- [ ] Output format specified
- [ ] Edge cases considered
- [ ] Tested with multiple inputs
- [ ] Performance measured

## 🔗 LangChain Prompt Types

### PromptTemplate
Basic string template with variables.

### ChatPromptTemplate
Multi-message chat templates.

### FewShotPromptTemplate
Templates with example-based learning.

### MessagesPlaceholder
Dynamic message insertion in chat templates.

## 🐛 Common Issues

### Template Variable Errors
```python
# Ensure all variables are provided
template = PromptTemplate(
    input_variables=["var1", "var2"],
    template="{var1} and {var2}"
)
# Must provide both var1 and var2
```

### Streamlit Port Conflicts
```bash
# Use a different port if 8501 is busy
streamlit run Prompt/prompt_ui.py --server.port 8502
```

## 📚 Resources

- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🔗 Related Directories

- `../chatbot/`: Advanced chatbot with prompt templates
- `../2.ChatModels/`: Chat model implementations
- `../1.LLMS/`: Basic LLM examples

---

**Next Steps**: Explore the `chatbot` directory for complete chatbot implementations using these prompt techniques.