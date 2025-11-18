# API Reference

This document provides detailed information about the AmbedkarGPT API.

## Core Classes

### AmbedkarGPT

The main class that implements the RAG pipeline.

```python
class AmbedkarGPT:
    def __init__(self, config: Config = Config()):
        """Initialize the RAG system with configuration."""
        
    def setup(self) -> bool:
        """Set up the pipeline components."""
        
    def ask(self, question: str) -> Tuple[str, List[Document]]:
        """Ask a question and get a response."""
```

### Config

Configuration class for the RAG system.

```python
class Config:
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "mistral:latest"
    # ... other configuration options
```

## Main Methods

### Process Documents

```python
def process_documents(directory: str) -> List[Document]:
    """Process all documents in the given directory."""
```

### Generate Embeddings

```python
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of text chunks."""
```

### Query Processing

```python
def process_query(query: str, k: int = 4) -> Dict:
    """Process a query and return relevant documents."""
```

## Response Format

API responses follow this structure:

```python
{
    "answer": "The generated answer",
    "sources": [
        {
            "content": "Document content...",
            "metadata": {
                "source": "filename.txt",
                "page": 1
            }
        }
    ],
    "confidence": 0.85
}
```

## Error Handling

Common exceptions:

- `DocumentLoadError`: Failed to load documents
- `EmbeddingError`: Error generating embeddings
- `LLMError`: Error from the language model
- `VectorStoreError`: Error accessing the vector database

## Example Usage

```python
from ambedkargpt import AmbedkarGPT, Config

# Initialize with custom config
config = Config(
    chunk_size=1000,
    llm_model="mistral:latest"
)

# Create instance
rag = AmbedkarGPT(config)

# Set up the pipeline
if rag.setup():
    # Ask a question
    response, sources = rag.ask("What were Ambedkar's views on education?")
    print(f"Answer: {response}")
    print("Sources:", [s.metadata['source'] for s in sources])
```

For more advanced usage, see the [Extending Guide](./extending.md).
