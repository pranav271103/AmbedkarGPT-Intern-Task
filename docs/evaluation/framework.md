# Evaluation Framework

This document describes the evaluation framework used to assess AmbedkarGPT's performance.

## Overview

The evaluation framework measures the system's performance across multiple dimensions:

- **Retrieval Quality**: How well the system finds relevant information
- **Answer Quality**: How accurate and relevant the generated answers are
- **Efficiency**: How quickly the system processes queries

## Key Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| Hit Rate | Percentage of queries where correct document is in top-k results | 100% |
| MRR | Mean Reciprocal Rank of first relevant document | 1.0 |
| Precision@k | Precision of top-k retrieved documents | 1.0 |
| Faithfulness | How well the answer is grounded in the context | 1.0 |
| Answer Relevance | Semantic similarity between question and answer | 1.0 |
| ROUGE-L | Text overlap with reference answers | 1.0 |
| BLEU | N-gram overlap with reference answers | 1.0 |
| Cosine Similarity | Embedding-based similarity with references | 1.0 |

## Running Evaluations

1. Prepare your test dataset in `test_dataset.json`:
   ```json
   [
     {
       "id": "q1",
       "question": "What were Ambedkar's views on education?",
       "ground_truth": "Ambedkar believed education was crucial for social transformation...",
       "source_documents": ["speech1.txt", "essay3.txt"]
     }
   ]
   ```

2. Run the evaluation script:
   ```bash
   python evaluation.py
   ```

3. View results in `test_results.json`

## Interpreting Results

- **Hit Rate > 0.8**: Excellent retrieval performance
- **MRR > 0.7**: Good ranking of relevant documents
- **ROUGE-L > 0.5**: Good content coverage in answers
- **Faithfulness > 0.8**: Answers are well-grounded in sources

## Customizing the Framework

### Adding New Metrics

1. Add your metric to `MetricsCalculator` class
2. Update the evaluation loop to compute the metric
3. Add it to the results dictionary

### Changing Evaluation Parameters

Modify the `EvaluationConfig` class:

```python
class EvaluationConfig:
    # Number of documents to retrieve
    RETRIEVER_K = 4
    
    # Chunking strategies to evaluate
    CHUNK_STRATEGIES = {
        "small": {"chunk_size": 250, "overlap": 50},
        "medium": {"chunk_size": 500, "overlap": 100},
        "large": {"chunk_size": 1000, "overlap": 150}
    }
```

## Best Practices

1. **Test on Diverse Queries**: Include various question types to ensure comprehensive evaluation.
2. **Balance the Dataset**: Include both simple and complex queries for a thorough assessment.
3. **Monitor Performance**: Track metrics over time to identify trends and areas for improvement.
4. **Compare Strategies**: Test different retrieval and generation configurations to optimize performance.
5. **Document Findings**: Keep detailed records of evaluation results and any changes made to the system.
