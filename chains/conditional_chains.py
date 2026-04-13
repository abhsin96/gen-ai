from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Clearify the sentiment of the following feedback text into prositive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction":parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positve feedback \n {feedback}",
    input_variables=["feedback"],
    
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"],
    
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "Negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({
    "feedback":"This is a amazing phone"
})

print(result)

# chain.get_graph().print_ascii()

#     +-------------+      
#     | PromptInput |      
#     +-------------+      
#             *            
#             *            
#             *            
#    +----------------+    
#    | PromptTemplate |    
#    +----------------+    
#             *            
#             *            
#             *            
#      +------------+      
#      | ChatOpenAI |      
#      +------------+      
#             *            
#             *            
#             *            
# +----------------------+ 
# | PydanticOutputParser | 
# +----------------------+ 
#             *            
#             *            
#             *            
#        +--------+        
#        | Branch |        
#        +--------+        
#             *            
#             *            
#             *            
#     +--------------+     
#     | BranchOutput |     
#     +--------------+ 