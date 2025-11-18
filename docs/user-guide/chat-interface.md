# Chat Interface Guide

This guide explains how to effectively use the AmbedkarGPT chat interface.

## Starting a Session

1. Launch the application:
   ```bash
   python main.py
   ```

2. You'll see the welcome message and system status:
   ```
   ======================================
   🤖 AmbedkarGPT - Ask about Dr. Ambedkar's Works
   ======================================
   
   [System] Initializing pipeline...
   [System] Loading documents...
   [System] Ready! Type your question or 'quit' to exit.
   ```

## Asking Questions

- Type your question and press Enter
- The system will process your query and provide an answer
- Each answer includes relevant source documents

### Example Session

```
Q: What were Dr. Ambedkar's views on education?

A: Dr. Ambedkar strongly believed that education was the most powerful tool for social transformation...

Sources:
- speech1.txt (pages 3-5)
- speech4.txt (page 12)

Q: 
```

## Available Commands

- `quit` or `exit` - End the session
- `clear` - Clear the screen
- `sources` - Show information about loaded documents

## Understanding the Output

Each response includes:

1. **Answer**: The generated response based on the documents
2. **Source Documents**: References to the original text used
3. **Confidence Score**: Indicates how confident the system is in the answer

## Tips for Better Results

1. **Be specific**: Instead of "Tell me about Ambedkar," ask "What were Ambedkar's views on social justice?"
2. **Use quotes**: For exact phrases, use quotation marks
3. **Ask follow-up questions**: The system maintains context within a session
4. **Check sources**: Always verify information from the provided sources

## Troubleshooting

If you encounter issues:
- Ensure the Ollama server is running
- Check that documents are in the `corpus/` directory
- Verify that the model is properly loaded

For more advanced usage, see the [Working with Documents](../user-guide/working-with-documents.md) guide.
