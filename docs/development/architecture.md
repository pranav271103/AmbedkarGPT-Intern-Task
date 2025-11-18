# System Architecture

This document provides an overview of AmbedkarGPT's architecture and design principles.

## Core Components

### 1. Document Processing Pipeline

- **Document Loader**: Loads documents from various formats
- **Text Splitter**: Chunks documents into manageable pieces
- **Embedding Model**: Converts text into vector representations
- **Vector Store**: Efficiently stores and retrieves document chunks

### 2. Query Processing

- **Query Understanding**: Processes and normalizes user queries
- **Vector Search**: Finds relevant document chunks
- **Context Assembly**: Combines relevant chunks into context

### 3. Generation

- **Prompt Engineering**: Constructs effective prompts for the LLM
- **LLM Interface**: Handles communication with the language model
- **Response Generation**: Formats and returns responses

## Data Flow

1. **Ingestion Phase**:
   - Documents are loaded and split into chunks
   - Chunks are converted to embeddings
   - Embeddings are stored in the vector database

2. **Query Phase**:
   - User query is converted to an embedding
   - Similar document chunks are retrieved
   - Context is assembled and sent to the LLM
   - Response is generated and returned

## Configuration

Key configuration parameters:

```python
class Config:
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Embeddings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE = "cpu"  # or "cuda" for GPU
    
    # LLM
    LLM_MODEL = "mistral:latest"
    TEMPERATURE = 0.1
    MAX_TOKENS = 1000
    
    # Retrieval
    RETRIEVER_K = 4  # Number of chunks to retrieve
```

## Performance Considerations

- **CPU vs GPU**: GPU significantly speeds up embedding generation
- **Chunk Size**: Affects both retrieval quality and processing time
- **Batch Processing**: Process documents in batches for large corpora
- **Caching**: Vector store caches embeddings for faster access

## Extensibility

The system is designed to be easily extended:

1. **New Document Types**: Implement new document loaders
2. **Embedding Models**: Swap in different embedding models
3. **Vector Stores**: Support for different vector databases
4. **LLM Providers**: Easily switch between different LLM providers

For API reference, see the [API Documentation](./api.md).
