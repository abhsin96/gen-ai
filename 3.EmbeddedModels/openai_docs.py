from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

docs = [
    "The softmax function is widely used in machine learning (especially in classification problems) to convert a vector of raw scores (logits) into probabilities.",
    "Takes any real-valued numbers (positive, negative, large, small)",
    "Converts them into values between 0 and 1",
    "Ensures all outputs sum to 1 → behaves like probabilities",
    "Exponentiation makes larger values stand out more",
    "Normalization (division by sum) ensures probabilities sum to 1",
    "Smooth and differentiable → useful for training neural networks"

]

result = embedding.embed_documents(docs)

print(result)