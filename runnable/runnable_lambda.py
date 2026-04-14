# ============================================================================
# RunnableLambda Example - Custom Function Integration in LangChain
# ============================================================================
# This script demonstrates how to use RunnableLambda to integrate custom
# Python functions into LangChain chains. RunnableLambda allows you to wrap
# any Python function and use it as part of a chain execution.
# ============================================================================

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda


load_dotenv()
model = ChatOpenAI()
parser = StrOutputParser()

def word_count(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template="Generate a interesting facts about {topic}",
    input_variables=["topic"]  # Specifies that 'topic' is a required input variable
)

# ============================================================================
# Chain Construction
# ============================================================================

# Chain 1: Sequential chain that processes the prompt through model and parser
# Flow: PromptTemplate → ChatOpenAI → StrOutputParser
# This chain takes a topic, generates facts, and returns them as a string
chain1 = RunnableSequence(prompt1, model, parser)

# Parallel Chain: Executes multiple operations simultaneously on the same input
# This demonstrates the power of RunnableLambda for custom processing
parallel_chain = RunnableParallel({
    # "fact": Passes the generated fact text through unchanged using RunnablePassthrough
    "fact": RunnablePassthrough(),
    
    # "length": Uses RunnableLambda to wrap our custom word_count function
    # RunnableLambda allows any Python function to be used in a chain
    # This will count the words in the generated fact
    "length": RunnableLambda(word_count)
})
# Output will be a dictionary: {"fact": "<generated text>", "length": <word count>}

# ============================================================================
# Final Chain Assembly
# ============================================================================

# Combine both chains sequentially:
# 1. chain1 generates facts about the topic
# 2. parallel_chain processes the fact in parallel:
#    - Preserves the original fact text
#    - Calculates word count using our custom function via RunnableLambda
final_chain = RunnableSequence(chain1, parallel_chain)

# ============================================================================
# Chain Execution
# ============================================================================

# Invoke the complete chain with the topic "IPL" (Indian Premier League)
# Execution flow:
# 1. Topic "IPL" → prompt1 → "Generate a interesting facts about IPL"
# 2. Prompt → model → AI-generated facts about IPL
# 3. Model output → parser → Plain text string
# 4. Text → parallel_chain:
#    - RunnablePassthrough: Preserves the fact text
#    - RunnableLambda(word_count): Counts words in the fact
# 5. Returns: {"fact": "<IPL facts>", "length": <number of words>}
result = final_chain.invoke({
    "topic": "IPL"
})

# Display the final result containing both the fact and its word count
print(result)

#               +-------------+                
#               | PromptInput |                
#               +-------------+                
#                       *                      
#                       *                      
#                       *                      
#              +----------------+              
#              | PromptTemplate |              
#              +----------------+              
#                       *                      
#                       *                      
#                       *                      
#                +------------+                
#                | ChatOpenAI |                
#                +------------+                
#                       *                      
#                       *                      
#                       *                      
#             +-----------------+              
#             | StrOutputParser |              
#             +-----------------+              
#                       *                      
#                       *                      
#                       *                      
#        +----------------------------+        
#        | Parallel<fact,length>Input |        
#        +----------------------------+        
#               **            ***              
#             **                 **            
#           **                     **          
# +-------------+              +------------+  
# | Passthrough |              | word_count |  
# +-------------+              +------------+  
#               **            ***              
#                 **        **                 
#                   **    **                   
#       +-----------------------------+        
#       | Parallel<fact,length>Output |        
#       +-----------------------------+  