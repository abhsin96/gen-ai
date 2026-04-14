from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a interesting facts about \n {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {topic}",
    input_variables=["topic"]
)

gen_facts_chain = RunnableSequence(prompt1, model, parser)

branch = RunnableBranch(
    (lambda x: len(x.split()) > 100 , RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(gen_facts_chain,branch)

result = final_chain.invoke("Batman")

print(result)

#   +-------------+    
#   | PromptInput |    
#   +-------------+    
#           *          
#           *          
#           *          
# +----------------+   
# | PromptTemplate |   
# +----------------+   
#           *          
#           *          
#           *          
#   +------------+     
#   | ChatOpenAI |     
#   +------------+     
#           *          
#           *          
#           *          
# +-----------------+  
# | StrOutputParser |  
# +-----------------+  
#           *          
#           *          
#           *          
#     +--------+       
#     | Branch |       
#     +--------+       
#           *          
#           *          
#           *          
#   +--------------+   
#   | BranchOutput |   
#   +--------------+   