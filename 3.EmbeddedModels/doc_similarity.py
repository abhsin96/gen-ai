from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "India has produced many legendary cricketers who have shaped the game globally. Sachin Tendulkar, often called the “God of Cricket,” set countless records and inspired generations with his consistency and longevity.",
    "Virat Kohli is known for his aggressive batting style and remarkable fitness, making him one of the best modern players.",
    "MS Dhoni stands out as a calm and strategic captain who led India to multiple ICC trophies.",
    "Rohit Sharma is famous for his elegant stroke play and record-breaking double centuries in ODIs.",
    "Kapil Dev remains iconic for leading India to its first World Cup victory in 1983 and for his all-round brilliance."
]

query ="tell me about kapil"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity between query and documents
similarities = cosine_similarity([query_embedding], doc_embedding)[0]

index, similarity =(sorted(list(enumerate(similarities)), key= lambda x:x[1])[-1])

print(query)
print(documents[index])
print("similarity {:.2f}".format(similarity))
