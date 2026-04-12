from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Using a supported model that works with HuggingFace serverless inference

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

# Initialize StringOutputParser to convert LLM output to clean string
string_parser = StrOutputParser()

# 1 Prompt -> detailed prompt (intuitive and comprehensive)
detailed_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable expert who explains topics clearly and thoroughly. "
               "When answering, organize your response into these sections:\n"
               "1. What it is (simple definition)\n"
               "2. Why it matters (importance and benefits)\n"
               "3. How it works (key features and concepts)\n"
               "4. Real-world examples (practical applications)"),
    ("user", "Tell me everything important about {topic}. Explain it in a way that's easy to understand but still detailed.")
])

# 2 Prompt -> summary prompt (intuitive and focused)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a master at distilling information into its essence. "
               "Create summaries that capture the main points clearly and concisely. "
               "Use exactly 5 sentences to cover: what, why, how, benefits, and key takeaway."),
    ("user", "Read this text and give me the 5 most important points in simple, clear sentences:\n\n{text}")
])

# Create a SINGLE unified chain using LCEL (LangChain Expression Language)
# This chains detailed_prompt -> model -> string_parser -> summary_prompt -> model -> string_parser
from langchain_core.runnables import RunnablePassthrough

# Combined chain: detailed response flows directly into summary
combined_chain = (
    {"topic": RunnablePassthrough()}  # Pass topic to detailed_prompt
    | detailed_prompt                  # Generate detailed response
    | model                             # LLM processes detailed prompt
    | string_parser                     # Convert to clean string
    | (lambda detailed_text: {"text": detailed_text})  # Transform output for summary_prompt
    | summary_prompt                    # Generate summary from detailed response
    | model                             # LLM processes summary prompt
    | string_parser                     # Convert final output to clean string
)

# Example usage:
if __name__ == "__main__":
    # Single chain invocation - automatically gets detailed response AND summary
    topic = "One piece"
    
    # Option 1: Get only the final summary (one-liner!)
    final_summary = combined_chain.invoke(topic)
    print("=== FINAL SUMMARY (5 sentences) ===")
    print(final_summary)
    
    # Option 2: If you still want both detailed and summary separately:
    # Keep the original two chains for flexibility
    detailed_chain = detailed_prompt | model | string_parser
    summary_chain = summary_prompt | model | string_parser