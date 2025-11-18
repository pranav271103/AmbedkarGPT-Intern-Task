# Extending AmbedkarGPT

This guide explains how to extend AmbedkarGPT with custom components.

## Custom Document Loaders

To add support for new document types, create a class that implements the `BaseLoader` interface:

```python
from typing import List
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader

class CustomLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        # Implement loading logic here
        return [Document(page_content=content, metadata={"source": self.file_path})]
```

## Custom Embedding Models

To use a different embedding model:

```python
from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Implement batch embedding
        pass
        
    def embed_query(self, text: str) -> List[float]:
        # Implement single text embedding
        pass
```

## Custom LLM Integration

To integrate a different LLM provider:

```python
from langchain.llms.base import LLM

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop=None) -> str:
        # Implement LLM call
        return "Generated response"
```

## Custom Retrieval Strategies

Implement a custom retriever:

```python
from typing import List
from langchain.schema import BaseRetriever, Document

class CustomRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Implement retrieval logic
        return []
```

## Adding New Metrics

To add evaluation metrics:

```python
class CustomMetrics:
    @staticmethod
    def calculate_metric(generated: str, reference: str) -> float:
        # Implement custom metric
        return 0.0
```

## Plugin System

AmbedkarGPT supports a plugin system for easy extension:

1. Create a Python package with this structure:
   ```
   ambedkargpt_plugin/
   ├── __init__.py
   ├── loaders.py
   ├── embeddings.py
   └── llms.py
   ```

2. Register your plugin in `setup.py`:
   ```python
   entry_points={
       'ambedkargpt.plugins': [
           'custom = ambedkargpt_plugin',
       ],
   }
   ```

## Best Practices

1. **Error Handling**: Always include proper error handling
2. **Logging**: Use Python's logging module
3. **Testing**: Write unit tests for your extensions
4. **Documentation**: Document your extensions thoroughly

## Example: Adding a New File Format

1. Create a new loader in `loaders/`
2. Update the document processor to use your loader
3. Add tests in `tests/`
4. Update the documentation

For more examples, check the [examples](https://github.com/pranav27103/AmbedkarGPT-Intern-Task/examples) directory.
