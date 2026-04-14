from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))

loader = DirectoryLoader(
    path=f"{script_dir}/book",  # f-string to substitute script_dir variable
    glob="*.pdf",  # Pattern to match PDF files (note: should be *.pdf not .pdf)
    loader_cls=PyPDFLoader  # Use PyPDFLoader for PDF files
)

doc = loader.lazy_load()

print(doc)