from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

# First prompt: Generate facts about a topic
prompt1 = PromptTemplate(
    template="Generate a interesting facts about {topic}",
    input_variables=["topic"]
)

# Second prompt: Summarize and enhance the facts
prompt2 = PromptTemplate(
    template="Take these facts and create a brief, engaging summary:\n\n{facts}",
    input_variables=["facts"]
)

# Create a sequence with 2 prompts
chain1 = RunnableSequence(prompt1, model, parser,prompt2, model, parser)

result = chain1.invoke({
    "topic": "Laughing"
})

print(result)