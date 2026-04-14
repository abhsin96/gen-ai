from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import  StrOutputParser
import os

load_dotenv()

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to cricket.txt
file_path = os.path.join(script_dir, "cricket.txt")

loader = TextLoader(file_path, encoding="utf-8")

doc = loader.load()

model = ChatOpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write summary of the poem \n {docs}",
    input_variables=["docs"]
)

chain = prompt | model | parser

result = chain.invoke({
    "docs":doc[0].page_content
})

print(result)
