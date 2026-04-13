from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()
model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz in to a single document \n notes-> {notes} and quiz -> {quiz}",
    input_variables=["notes","quiz"]
)

parser = StrOutputParser()


parallel_chains = RunnableParallel({
    "notes": prompt1|model1|parser,
    "quiz": prompt2|model2|parser,
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chains | merge_chain

text = """
The field of Natural Language Processing (NLP) is concerned with how machines understand, interpret, generate and interact with natural languages – i.e., languages as spoken and written by humans such as English – and how they are processed. Since State-Of-The-Art (SOTA) LLMs are trained on massive amounts of text, they are not only able to handle a variety of different natural languages, but also programming languages like C++ or Python.
The flagship models, which typically consist of several hundreds to thousands of billions of parameters, can usually not be run locally, but are only available via an online interface or API like OpenAI’s GPT models [8], Google Gemini [9] (previously known as Google Bard), as well as the Claude 3 model family [10]. In contrast, many of the open-source models like Mistral [11] and its code-generating variant Codestral, are relatively small models that can be run locally on consumer GPUs. The model family Phi [12] by Microsoft is developed with the goal of providing small language models to be run locally, including on smartphones. The open-weight Llama models, developed by Meta, cover a large range of pre-trained sizes, from 8 B to 405 B (B = billion parameters). The Llama 3 [13] weights of the models are available to run locally, but details on training algorithm and data are not fully disclosed. For research, being able to access the model’s weights and settings is an advantage over closed models due to reproducibility, transparency and size optimizations using quantization techniques. As running LLMs is not only compute-intensive but also requires large GPU memory, quantization techniques are commonly applied to language models after training [14], where parameters of reduced precision are used rather than 32-bit full precision (floating-point) values. It has been found that even 4-bit quantized models perform well [15], thus enabling larger LLMs to fit into the same memory using lower-bit quantization which can be advantageous.
Assessing the performance of models is a challenging task, as natural languages are ambiguous and vague [16]: two sentences can differ at a word level but still be semantically similar. Originally proposed by Alan Turing in 1950 to distinguish humans from machines, the Turing test – recently reported as passed by modern LLMs with up to 73 % human deception rate [17] – involves subjective human judgment, motivating the development of diverse datasets and quantitative evaluation methods for NLP. Early methods like BLEU [18], used for evaluation of machine-translation, and ROUGE [19], applied to automatic summarization, relied on word overlap and thus do not capture semantic understanding. This led to the development of more sophisticated benchmarks based on specific tasks like question answering. A landmark in this effort is the TyDi QA dataset [20], which was designed to test language understanding across 11 typologically diverse languages, moving beyond English-centric evaluation. To more comprehensively test syntax, semantic similarity and reasoning, modern benchmarks like General Language Understanding Evaluation (GLUE) [21], SWAG: a dataset for grounded commonsense inference [22], and the Massive Multitask Language Understanding (MMLU) dataset [23] were developed as well as datasets for specific fields or tasks like MATH [24] and IFEval [25]. The development of increasingly capable LLMs and their application to new problem domains have significantly raised the difficulty of evaluation. As a result, many benchmark datasets have been revised or extended to remain challenging [26], [27], [28]. In the domain of code generation, specialized benchmarks have been introduced where models are tasked with generating code from natural language specifications, and correctness is evaluated using test cases [29], [30].
To test the LLM knowledge and reasoning capabilities, the Google-Proof Q&A GPQA [31] and most recently superGPQA [32] are designed to evaluate graduate-level knowledge and reasoning capabilities across many disciplines, with the superGPQA reporting an accuracy of 
 for the DeepSeek-R1 [33] reasoning model. Although benchmarks include mechanical engineering questions, they are usually underrepresented, e.g., with only 
 in Humanity’s Last Exam (HLE) [34]. Moreover, despite reports of strong performance on multiple choice mechanical engineering exam questions [35], this study only finds a low correlation between common benchmarks and correct simulation code for the presented mechanical engineering problems.
Although LLMs are not inherently well-suited for solving computational problems – a well-known example is that many LLMs fail to correctly count the number of occurrences of “r” in “strawberry” – they handle many programming languages [36], tools and APIs [37], [38] well, and they can even act as dynamic agents [39]. To provide LLMs with up-to-date knowledge and improve domain knowledge, Retrieval Augmented Generation (RAG) can be used [40], potentially also for unknown, closed source code [41]. Alternatively, pre-trained models like Llama can be fine-tuned for specific domains and simulation tools, as previously done for Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS) [42], but a (high-quality) dataset and large amount of computing power are required. Commonly a ground truth is provided for evaluating performance [43]. Alternatives are one or more LLMs working together [44], where roles such as worker and evaluator, sometimes referred to as judge, might be assigned both for general answers [45] and engineering specific tasks [42], [46]. As for many applications not only plain programming languages are relevant, more recently specific benchmarks for agents to test function calling [47] and tool usage for real-world domains [48] were developed. Traditional supervised pre-training does typically not account for the multi-step dynamic environments the agents are interacting with. To address this, [49] performs iterative fine-tuning with an LLM as a judge.
"""


result = chain.invoke({
    "text":text
})

print(result)

chain.get_graph().print_ascii()



#   +---------------------------+            
#             | Parallel<notes,quiz>Input |            
#             +---------------------------+            
#                  **               **                 
#               ***                   ***              
#             **                         **            
# +----------------+                +----------------+ 
# | PromptTemplate |                | PromptTemplate | 
# +----------------+                +----------------+ 
#           *                               *          
#           *                               *          
#           *                               *          
#   +------------+                    +------------+   
#   | ChatOpenAI |                    | ChatOpenAI |   
#   +------------+                    +------------+   
#           *                               *          
#           *                               *          
#           *                               *          
# +-----------------+              +-----------------+ 
# | StrOutputParser |              | StrOutputParser | 
# +-----------------+              +-----------------+ 
#                  **               **                 
#                    ***         ***                   
#                       **     **                      
#            +----------------------------+            
#            | Parallel<notes,quiz>Output |            
#            +----------------------------+            
#                           *                          
#                           *                          
#                           *                          
#                  +----------------+                  
#                  | PromptTemplate |                  
#                  +----------------+                  
#                           *                          
#                           *                          
#                           *                          
#                    +------------+                    
#                    | ChatOpenAI |                    
#                    +------------+                    
#                           *                          
#                           *                          
#                           *                          
#                 +-----------------+                  
#                 | StrOutputParser |                  
#                 +-----------------+                  
#                           *                          
#                           *                          
#                           *                          
#               +-----------------------+              
#               | StrOutputParserOutput |              
#               +-----------------------+          
