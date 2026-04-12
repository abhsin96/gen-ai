from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

# Using a supported model that works with HuggingFace serverless inference

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

# Define Pydantic model for structured output (Modern approach)
class PersonInfo(BaseModel):
    """Information about a person extracted from text."""
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")

# Create Pydantic output parser
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# Get format instructions for the prompt
format_instructions = parser.get_format_instructions()

# Create prompt template with format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that extracts structured information from text."),
    ("user", "Extract the name and age of the person in the following text:\n\n{text}\n\n{format_instructions}")
])

# Create chain using LCEL
chain = prompt | model | parser

# Invoke the chain
result = chain.invoke({
    "text": "John is 20 years old",
    "format_instructions": format_instructions
})

print(f"Parsed Result: {result}")
print(f"Name: {result.name}")
print(f"Age: {result.age}")