"""Research Paper Summarization Tool using Streamlit and LangChain."""

from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Constants
RESEARCH_PAPERS = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
]

EXPLANATION_STYLES = [
    "Beginner-Friendly",
    "Technical",
    "Code-Oriented",
    "Mathematical"
]

EXPLANATION_LENGTHS = [
    "Short (1-2 paragraphs)",
    "Medium (3-5 paragraphs)",
    "Long (detailed explanation)"
]



# Initialize model
model = ChatOpenAI()

# Streamlit UI
st.header("Research Paper Summarization Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    RESEARCH_PAPERS
)

style_input = st.selectbox(
    "Select Explanation Style",
    EXPLANATION_STYLES
)

length_input = st.selectbox(
    "Select Explanation Length",
    EXPLANATION_LENGTHS
)

template = load_prompt("template.json")

# Generate summary on button click
if st.button("Summarize"):
    prompt = template.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    
    result = model.invoke(prompt)
    st.write(result.content)