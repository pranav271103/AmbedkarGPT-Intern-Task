# Quick Start Guide

This guide will help you get started with AmbedkarGPT quickly.

## First Run

1. Start the Ollama server in a separate terminal:
   ```bash
   ollama serve
   ```

2. In another terminal, navigate to the project directory and run:
   ```bash
   python main.py
   ```

3. The application will:
   - Initialize the RAG pipeline
   - Process documents in the `corpus/` directory
   - Start the interactive chat interface

## Basic Usage

### Interactive Chat

After starting the application, you can ask questions about Dr. Ambedkar's works:

```
Q: What were Dr. Ambedkar's views on education?
```

The system will provide an answer based on the content in your corpus.

### Adding New Documents

1. Place new text files in the `corpus/` directory
2. Restart the application to process the new documents

### Configuration

Key configuration options can be found in `main.py`:

```python
class Config:
    CHUNK_SIZE = 500        # Size of text chunks
    CHUNK_OVERLAP = 50      # Overlap between chunks
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "mistral:latest"
```

## Example Queries

- "What were Dr. Ambedkar's main contributions to the Indian Constitution?"
- "Explain Dr. Ambedkar's views on social justice"
- "What did Ambedkar say about democracy in India?"

## Next Steps

- Learn more about [using the chat interface](../user-guide/chat-interface.md)
- Explore [advanced configuration options](../development/architecture.md#configuration)
- Understand how the [evaluation framework](../evaluation/framework.md) works
