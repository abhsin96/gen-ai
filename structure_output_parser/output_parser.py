from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
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

# 1 Prompt -> detailed prompt (optimized for comprehensive analysis)
detailed_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research assistant specializing in clear, comprehensive explanations. "
               "Structure your responses with: 1) Definition/Overview, 2) Key Concepts, 3) Real-world Examples, 4) Significance/Impact."),
    ("user", "Provide a detailed, well-structured report on {topic}. Include definitions, key concepts, examples, and practical implications.")
])

# 2 Prompt -> summary prompt (optimized for concise extraction)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a skilled summarization assistant. Extract and present only the most critical information in exactly 5 concise sentences. "
               "Prioritize key facts, main ideas, and actionable insights."),
    ("user", "Summarize the following text in exactly 5 clear, informative sentences:\n\n{text}")
])

# Example usage:
if __name__ == "__main__":
    # Using detailed prompt
    detailed_chain = detailed_prompt | model
    detailed_result = detailed_chain.invoke({"topic": "black hole"})
    print("Detailed Response:")
    print(detailed_result.content)
    print("\n" + "="*50 + "\n")
    
    # Using summary prompt
    summary_chain = summary_prompt | model
    summary_result = summary_chain.invoke({"text": detailed_result.content})
    print("Summary Response:")
    print(summary_result.content)