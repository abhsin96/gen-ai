from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

# Create two different prompts that will run in parallel
prompt1 = PromptTemplate.from_template(
    "Write a short poem about {topic}"
)

prompt2 = PromptTemplate.from_template(
    "Write a fun fact about {topic}"
)

# Create chains for each prompt
chain1 = prompt1 | model | parser
chain2 = prompt2 | model | parser

# Use RunnableParallel to execute both chains simultaneously
parallel_chain = RunnableParallel(
    poem=RunnableSequence(chain1),
    fact=RunnableSequence(chain2)
)

# Execute the parallel chain
if __name__ == "__main__":
    result = parallel_chain.invoke({"topic": "artificial intelligence"})
    
    print("=" * 50)
    print("POEM:")
    print("=" * 50)
    print(result["poem"])
    print("\n" + "=" * 50)
    print("FUN FACT:")
    print("=" * 50)
    print(result["fact"])
    print("\n" + "=" * 50)