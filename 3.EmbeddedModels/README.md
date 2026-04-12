# 3. Embedded Models (Text Embeddings)

This directory contains implementations for text embeddings and document similarity using various embedding models.

## 📋 Overview

Text embeddings convert text into numerical vector representations, enabling semantic search, similarity comparisons, and other advanced NLP tasks.

## 📁 Files

### `hf_local.py`
Local HuggingFace embedding model implementation.

**Features:**
- Uses `sentence-transformers/all-MiniLM-L6-v2` model
- Local execution without API calls
- Fast and efficient embeddings generation
- Privacy-focused implementation

**Model Details:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding Dimension: 384
- Performance: Balanced speed and quality
- Use Case: General-purpose embeddings

### `openai_docs.py`
OpenAI embeddings for document processing.

**Features:**
- OpenAI's text-embedding models
- High-quality embeddings
- Document indexing and storage
- Integration with vector databases

### `openai_query.py`
Query processing using OpenAI embeddings.

**Features:**
- Semantic search capabilities
- Query-document matching
- Similarity-based retrieval
- RAG (Retrieval-Augmented Generation) support

### `doc_similarity.py`
Document similarity comparison implementation.

**Features:**
- Compare multiple documents
- Calculate similarity scores
- Find related content
- Clustering and grouping

## 🚀 Getting Started

### Prerequisites

```bash
# Install required dependencies
pip install -r ../requirements.txt

# Additional dependencies for embeddings
pip install sentence-transformers
pip install langchain-huggingface
```

### Configuration

Update your `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
```

### Running Examples

```bash
# Local HuggingFace embeddings
python3 3.EmbeddedModels/hf_local.py

# OpenAI document embeddings
python3 3.EmbeddedModels/openai_docs.py

# OpenAI query processing
python3 3.EmbeddedModels/openai_query.py

# Document similarity
python3 3.EmbeddedModels/doc_similarity.py
```

## 💡 Use Cases

### 1. Semantic Search
Find documents based on meaning rather than exact keyword matches.

### 2. Document Clustering
Group similar documents together automatically.

### 3. Recommendation Systems
Recommend similar content based on user preferences.

### 4. Duplicate Detection
Identify duplicate or near-duplicate content.

### 5. RAG Applications
Retrieve relevant context for language model queries.

## 🔧 Embedding Models Comparison

| Model | Provider | Dimension | Speed | Quality | Cost |
|-------|----------|-----------|-------|---------|------|
| all-MiniLM-L6-v2 | HuggingFace | 384 | Fast | Good | Free |
| text-embedding-ada-002 | OpenAI | 1536 | Medium | Excellent | Paid |
| text-embedding-3-small | OpenAI | 1536 | Fast | Excellent | Paid |
| text-embedding-3-large | OpenAI | 3072 | Medium | Best | Paid |

## 📝 Example: Local Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings
text = "This is a sample document for embedding."
vector = embeddings.embed_query(text)

print(f"Embedding dimension: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
```

## 📝 Example: Document Similarity

```python
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents to compare
doc1 = "The cat sits on the mat."
doc2 = "A feline rests on the rug."
doc3 = "Python is a programming language."

# Generate embeddings
vec1 = embeddings.embed_query(doc1)
vec2 = embeddings.embed_query(doc2)
vec3 = embeddings.embed_query(doc3)

# Calculate similarity
sim_1_2 = cosine_similarity([vec1], [vec2])[0][0]
sim_1_3 = cosine_similarity([vec1], [vec3])[0][0]

print(f"Similarity (doc1, doc2): {sim_1_2:.4f}")  # High similarity
print(f"Similarity (doc1, doc3): {sim_1_3:.4f}")  # Low similarity
```

## 🔍 Vector Databases

Embeddings are often stored in vector databases for efficient similarity search:

- **Chroma**: Lightweight, easy to use
- **Pinecone**: Managed service, scalable
- **Weaviate**: Open-source, feature-rich
- **FAISS**: Facebook's similarity search library
- **Milvus**: Highly scalable, production-ready

## 🐛 Troubleshooting

### Model Download Issues
```bash
# Manually download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Memory Issues
For large document sets, process in batches:
```python
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    embeddings.embed_documents(batch)
```

## 🔗 Related Directories

- `../2.ChatModels/`: Chat model implementations
- `../chatbot/`: Chatbot with RAG capabilities
- `../Prompt/`: Prompt engineering for better retrieval

---

**Next Steps**: Explore the `Prompt` directory for advanced prompt engineering techniques.