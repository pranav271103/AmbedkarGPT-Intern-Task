# Working with Documents

This guide explains how to manage and work with documents in AmbedkarGPT.

## Supported Formats

AmbedkarGPT supports the following document formats:

- Plain text (`.txt`)
- Markdown (`.md`)
- PDF (`.pdf`) - requires additional dependencies
- Word documents (`.docx`)

## Adding Documents

1. Place your documents in the `corpus/` directory
2. Supported filename formats:
   - `speech1.txt`
   - `essay_on_education.pdf`
   - `constitution_analysis.docx`

## Document Structure

For best results, structure your documents with clear sections:

```
Title: Speech on Education
Date: January 1, 1950

[Content begins...]
```

## Document Metadata

You can include metadata in your documents using YAML frontmatter:

```yaml
---
title: Annihilation of Caste
author: Dr. B.R. Ambedkar
date: 1936-05-31
source: Speech at Jat-Pat Todak Mandal
---

[Document content...]
```

## Document Processing

When you start AmbedkarGPT, it will:

1. Scan the `corpus/` directory for supported files
2. Split documents into chunks (configurable in `main.py`)
3. Generate embeddings for each chunk
4. Store the processed data in the vector database

## Updating Documents

To update documents:

1. Make your changes to the document
2. Delete the corresponding vector database files in `chroma_db/`
3. Restart the application

## Best Practices

1. **Keep documents focused**: Each document should cover a specific topic
2. **Use clear formatting**: Headers, paragraphs, and lists improve readability
3. **Include citations**: Always attribute sources properly
4. **Avoid complex layouts**: Simple text works best for processing

## Troubleshooting

### Document Not Loading
- Check file permissions
- Verify the file format is supported
- Check the application logs for errors

### Poor Quality Results
- Ensure text is clean and well-formatted
- Check for encoding issues (use UTF-8)
- Consider splitting large documents into smaller, focused ones

For advanced document processing options, see the [Development Guide](../development/architecture.md).
