from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "dl-curriculum.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Summarize this page:\n\n{page_content}\n\nSummary:",
    input_variables=["page_content"]
)

chain = prompt | model | parser

if documents:
    summary = chain.invoke({"page_content": documents[0].page_content})
    print(f"\nSummary: {summary}")

all_summaries = []
for i, doc in enumerate(documents[:3]):
    summary = chain.invoke({"page_content": doc.page_content})
    all_summaries.append({"page": i + 1, "summary": summary})

full_text = "\n\n".join([doc.page_content for doc in documents])
