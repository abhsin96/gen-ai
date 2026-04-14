from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

loader = DirectoryLoader(
    path=f"{script_dir}/book",  # f-string to substitute script_dir variable
    glob="*.pdf",  # Pattern to match PDF files (note: should be *.pdf not .pdf)
    loader_cls=PyPDFLoader  # Use PyPDFLoader for PDF files
)

doc = loader.lazy_load()

print(doc)