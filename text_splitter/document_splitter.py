"""Python Code Splitter Example using LangChain Document Splitter.

This module demonstrates how to split Python code into manageable chunks
using LangChain's PythonCodeTextSplitter for better processing and analysis.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


def split_python_code(code: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split Python code into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        code: Python code string to be split
        chunk_size: Maximum size of each chunk (default: 500)
        chunk_overlap: Number of characters to overlap between chunks (default: 50)
    
    Returns:
        List of code chunks as strings
    
    Raises:
        ValueError: If code is empty or chunk_size is invalid
    """
    if not code or not code.strip():
        raise ValueError("Code string cannot be empty")
    
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    
    # Use RecursiveCharacterTextSplitter with Python-specific separators
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = python_splitter.split_text(code)
    return chunks


if __name__ == "__main__":
    # Example Python code to split
    sample_python_code = '''
import os
from typing import List, Dict, Optional

class DataProcessor:
    """A class to process and analyze data."""
    
    def __init__(self, data_path: str) -> None:
        """Initialize the DataProcessor.
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.data: Optional[List[Dict]] = None
    
    def load_data(self) -> None:
        """Load data from the specified path."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Simulated data loading
        self.data = [{"id": i, "value": i * 10} for i in range(100)]
        print(f"Loaded {len(self.data)} records")
    
    def process_data(self, filter_threshold: int = 50) -> List[Dict]:
        """Process and filter data based on threshold.
        
        Args:
            filter_threshold: Minimum value threshold for filtering
        
        Returns:
            Filtered list of data records
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        filtered_data = [
            record for record in self.data 
            if record["value"] >= filter_threshold
        ]
        return filtered_data
    
    def analyze_data(self) -> Dict[str, float]:
        """Analyze data and return statistics.
        
        Returns:
            Dictionary containing statistical metrics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        values = [record["value"] for record in self.data]
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }

def main() -> None:
    """Main function to demonstrate data processing."""
    processor = DataProcessor("data.json")
    try:
        processor.load_data()
        filtered = processor.process_data(filter_threshold=100)
        stats = processor.analyze_data()
        print(f"Statistics: {stats}")
        print(f"Filtered records: {len(filtered)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
'''
    
    # Split the Python code into chunks
    try:
        code_chunks = split_python_code(
            code=sample_python_code,
            chunk_size=300,
            chunk_overlap=50
        )
        
        print(f"\n{'='*60}")
        print("Python Code Splitter Example")
        print(f"{'='*60}\n")
        print(f"Total chunks created: {len(code_chunks)}\n")
        
        for idx, chunk in enumerate(code_chunks, 1):
            print(f"--- Chunk {idx} ({len(chunk)} characters) ---")
            print(chunk)
            print(f"\n{'.'*60}\n")
            
    except ValueError as e:
        print(f"Error splitting code: {e}")
