from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the embedding model with sentence-transformers/all-MiniLM-L6-v2
# This model will run locally on your machine
model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU available
    encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better similarity search
)

# Example: Generate embeddings for a single text
text = "This is a sample sentence to generate embeddings."
embedding_vector = embeddings.embed_query(text)

print(f"Model: {model_name}")
print(f"Text: {text}")
print(f"Embedding dimension: {len(embedding_vector)}")
print(f"Vertor values: {embedding_vector}")

# Example: Generate embeddings for multiple documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand human language."
]

doc_embeddings = embeddings.embed_documents(documents)

print(f"\nGenerated embeddings for {len(documents)} documents")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc[:50]}... -> Embedding shape: {len(doc_embeddings[i])}")
