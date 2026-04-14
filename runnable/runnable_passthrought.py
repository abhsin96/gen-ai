# ============================================================================
# RunnablePassthrough Example with LangChain
# ============================================================================
# This script demonstrates the use of RunnablePassthrough in LangChain to
# preserve intermediate outputs while continuing processing in parallel chains.
# 
# Key Concepts:
# 1. RunnablePassthrough: Passes input data through unchanged while allowing
#    other operations to run in parallel
# 2. RunnableParallel: Executes multiple runnables concurrently on the same input
# 3. RunnableSequence: Chains multiple runnables where output of one becomes
#    input to the next
# ============================================================================

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel

# Load environment variables (e.g., OPENAI_API_KEY) from .env file
load_dotenv()

# ============================================================================
# Initialize Components
# ============================================================================

# Initialize the ChatOpenAI model for generating responses
model = ChatOpenAI()

# String output parser to convert model responses to plain strings
parser = StrOutputParser()

# ============================================================================
# Define Prompt Templates
# ============================================================================

# First prompt: Generate facts about a topic
# This prompt takes a topic and generates interesting facts about it
prompt1 = PromptTemplate(
    template="Generate a interesting facts about {topic}",
    input_variables=["topic"]  # Expects 'topic' as input variable
)

# Second prompt: Summarize and enhance the facts
# This prompt takes the generated facts and creates a brief summary
prompt2 = PromptTemplate(
    template="Take these facts and create a brief, engaging summary:\n\n{facts}",
    input_variables=["facts"]  # Expects 'facts' as input variable
)

# ============================================================================
# Build Chain Components
# ============================================================================

# Create a sequence to generate facts:
# Input (topic) → prompt1 → model → parser → Output (facts as string)
fact = RunnableSequence(prompt1, model, parser)

# ============================================================================
# RunnableParallel with RunnablePassthrough
# ============================================================================
# This is the KEY part demonstrating RunnablePassthrough:
# 
# The parallel runnable creates a dictionary with two keys:
# 1. "fact": Uses RunnablePassthrough() to preserve the original facts
#    generated from the first sequence WITHOUT any modification
# 2. "summary": Processes the facts through prompt2 → model → parser
#    to create a summary
# 
# Why use RunnablePassthrough?
# - It allows you to keep the original data while also processing it
# - Useful when you need both raw and processed versions of data
# - Enables parallel processing without losing intermediate results
# ============================================================================
parallel = RunnableParallel({
    "fact": RunnablePassthrough(),  # Passes facts through unchanged
    "summary": RunnableSequence(prompt2, model, parser)  # Creates summary from facts
})

# ============================================================================
# Final Chain Assembly
# ============================================================================
# Complete chain flow:
# 1. Input: {"topic": "India"}
# 2. fact sequence generates facts about India
# 3. parallel runnable receives the facts and:
#    - Stores original facts in "fact" key (via RunnablePassthrough)
#    - Generates summary in "summary" key (via prompt2 sequence)
# 4. Output: {"fact": "<original facts>", "summary": "<summarized facts>"}
final_chain = RunnableSequence(fact, parallel)

# ============================================================================
# Execute the Chain
# ============================================================================
# Invoke the chain with a topic
result = final_chain.invoke({
    "topic": "India"
})

# Print the result containing both original facts and summary
# Expected output format:
# {
#   "fact": "<interesting facts about India>",
#   "summary": "<brief engaging summary of the facts>"
# }
print(result)