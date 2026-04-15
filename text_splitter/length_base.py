from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Get the project root directory (parent of text_splitter directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
# PDF file is located in document_loader directory
file_path = os.path.join(project_root, "document_loader", "dl-curriculum.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
     chunk_size=100, chunk_overlap=0
)

document = """If you’re reading this article, you probably have already heard about large language models (LLMs). Who hasn’t? In the end, LLMs are behind the super popular tools fueling the ongoing generative AI revolution, including ChatGPT, Google Bard, and DALL-E.

To deliver their magic, these tools rely on a powerful technology that allows them to process data and generate accurate content in response to the question prompted by the user. This is where LLMs kick in.

This article aims to introduce you to LLMs. After reading the following sections, we will know what LLMs are, how they work, the different types of LLMs with examples, as well as their advantages and limitations.

For newcomers to the subject, our Large Language Models (LLMs) Concepts Course is a perfect place to get a deep overview of LLMs. However, if you’re already familiar with LLM and want to go a step further by learning how to build LLM-power applications, check out our article How to Build LLM Applications with LangChain.

Let’s get started!"""

# texts = text_splitter.split_text(document)
texts = text_splitter.split_documents(documents)

splitter2 =  RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)

text2 = splitter2.split_text(document)

# print(texts[0].page_content)
print(text2)