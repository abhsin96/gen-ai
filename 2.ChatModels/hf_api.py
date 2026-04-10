from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Using a publicly available model that's supported by HuggingFace Serverless Inference
# Alternative options:
# - "mistralai/Mistral-7B-Instruct-v0.2"
# - "meta-llama/Llama-3.2-3B-Instruct" (requires access token)
# - "HuggingFaceH4/zephyr-7b-beta"
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is llm?")
print(result)