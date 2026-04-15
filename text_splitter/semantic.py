"""Semantic Text Splitter Example using LangChain.

This module demonstrates how to split text semantically using embeddings
to create meaningful chunks based on semantic similarity rather than just
character count or arbitrary delimiters.
"""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def split_text_semantically(
    text: str,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: Optional[float] = None,
    number_of_chunks: Optional[int] = None
) -> List[str]:
    """Split text into semantically meaningful chunks using embeddings.
    
    This function implements semantic chunking by analyzing the semantic similarity
    between sentences using embeddings. It creates breakpoints where the
    semantic similarity drops significantly, ensuring each chunk contains
    semantically related content.
    
    Note: SemanticChunker is not available in langchain_text_splitters v1.1.1.
    This implementation provides similar functionality using embeddings and
    similarity-based splitting.
    
    Args:
        text: The text string to be split into semantic chunks
        breakpoint_threshold_type: Method to determine breakpoints:
            - "percentile": Use percentile of similarity scores (default)
            - "standard_deviation": Use standard deviation from mean
            - "interquartile": Use interquartile range
        breakpoint_threshold_amount: Threshold value for the chosen type.
            For percentile: value between 0-100 (default: 95)
            For std_deviation: number of standard deviations (default: 3)
            For interquartile: multiplier for IQR (default: 1.5)
        number_of_chunks: If specified, splits into approximately this many chunks
    
    Returns:
        List of semantically coherent text chunks
    
    Raises:
        ValueError: If text is empty or API key is not configured
        RuntimeError: If embedding generation fails
    
    Example:
        >>> text = "Your long document here..."
        >>> chunks = split_text_semantically(text, "percentile", 90)
        >>> print(f"Created {len(chunks)} semantic chunks")
    """
    if not text or not text.strip():
        raise ValueError("Text string cannot be empty")
    
    try:
        # Initialize embeddings model
        embeddings = OpenAIEmbeddings()
        
        # Split text into sentences first
        sentence_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? ", "; "],
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        sentences = sentence_splitter.split_text(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Generate embeddings for each sentence
        sentence_embeddings = embeddings.embed_documents(sentences)
        
        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            sim = cosine_similarity(
                [sentence_embeddings[i]], 
                [sentence_embeddings[i + 1]]
            )[0][0]
            similarities.append(sim)
        
        # Determine breakpoints based on threshold type
        breakpoints = [0]
        
        if number_of_chunks:
            # Split into fixed number of chunks
            chunk_size = len(sentences) // number_of_chunks
            breakpoints = [i * chunk_size for i in range(number_of_chunks)]
            breakpoints.append(len(sentences))
        else:
            # Use similarity-based breakpoints
            if breakpoint_threshold_type == "percentile":
                threshold_value = breakpoint_threshold_amount or 95
                threshold = np.percentile(similarities, 100 - threshold_value)
            elif breakpoint_threshold_type == "standard_deviation":
                threshold_value = breakpoint_threshold_amount or 3
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                threshold = mean_sim - (threshold_value * std_sim)
            elif breakpoint_threshold_type == "interquartile":
                threshold_value = breakpoint_threshold_amount or 1.5
                q1 = np.percentile(similarities, 25)
                q3 = np.percentile(similarities, 75)
                iqr = q3 - q1
                threshold = q1 - (threshold_value * iqr)
            else:
                threshold = np.percentile(similarities, 5)  # Default
            
            # Find breakpoints where similarity drops below threshold
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    breakpoints.append(i + 1)
            breakpoints.append(len(sentences))
        
        # Create chunks based on breakpoints
        chunks = []
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            chunk_text = " ".join(sentences[start_idx:end_idx])
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks if chunks else [text]
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate semantic chunks: {str(e)}") from e


if __name__ == "__main__":
    sample_text = """
    Large Language Models (LLMs) have revolutionized natural language processing. 
    These models are trained on vast amounts of text data and can generate human-like responses. 
    GPT-4, Claude, and other modern LLMs use transformer architectures to process text.
    
    The training process for LLMs involves multiple stages. First, models undergo pre-training 
    on large corpora of text from the internet. Then, they are fine-tuned using supervised learning 
    with human feedback. This process is called Reinforcement Learning from Human Feedback (RLHF).
    
    Text splitting is crucial when working with LLMs. Documents often exceed the context window 
    limits of these models. Traditional methods split text by character count or fixed delimiters, 
    which can break semantic meaning. Semantic splitting preserves the coherence of ideas.
    
    Embeddings play a key role in semantic text splitting. By converting text into vector 
    representations, we can measure semantic similarity between sentences. When similarity drops 
    significantly, it indicates a good breakpoint for splitting. This ensures each chunk maintains 
    topical coherence.
    
    Applications of semantic splitting include retrieval-augmented generation (RAG), document 
    analysis, and content summarization. RAG systems benefit from semantically coherent chunks 
    because they improve retrieval accuracy. When chunks contain complete ideas, the LLM receives 
    better context for generating responses.
    
    The future of text processing will likely involve more sophisticated semantic understanding. 
    As embedding models improve, we'll see better chunk boundaries that respect discourse structure. 
    This will enable more accurate and contextually aware AI applications across various domains.
    """
    
    try:
        chunks_percentile = split_text_semantically(
            text=sample_text,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90
        )
        
        chunks_std = split_text_semantically(
            text=sample_text,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=2.0
        )
        
        chunks_fixed = split_text_semantically(
            text=sample_text,
            number_of_chunks=3
        )
            
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
