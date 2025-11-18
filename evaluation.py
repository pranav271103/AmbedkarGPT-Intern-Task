"""
AmbedkarGPT: Phase 2 - Evaluation
"""
import os
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

# CORRECTED IMPORTS for langchain 1.0.x + langchain-community 0.4.x
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

class EvaluationConfig:
    """Configuration for evaluation."""

    # CORRECTED PATHS - Match your actual file structure
    CORPUS_PATH = "./corpus"  # Load from corpus/ folder with 6 txt files
    TEST_DATASET_PATH = "./test_dataset.json"  # In main folder
    EVAL_DB_DIR = "./chroma_db_eval"
    RESULTS_FILE = "./test_results.json"

    # Models
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE = "cpu"
    LLM_MODEL = "mistral:latest"

    # Metrics
    ROUGE_TYPE = 'rougeL'
    RETRIEVER_K = 4

    # Chunking strategies
    CHUNK_STRATEGIES = {
        "small": {"chunk_size": 250, "chunk_overlap": 50},
        "medium": {"chunk_size": 500, "chunk_overlap": 100},
        "large": {"chunk_size": 900, "chunk_overlap": 150},
    }


class MetricsCalculator:
    """Calculates all evaluation metrics."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def hit_rate(self, retrieved_docs, expected_sources):
        """Did we retrieve ANY relevant document?"""
        if not expected_sources:
            return 1.0

        retrieved_sources = {Path(doc.metadata.get("source", "")).name for doc in retrieved_docs}
        expected_set = set(expected_sources)

        return 1.0 if any(s in retrieved_sources for s in expected_set) else 0.0

    def mrr(self, retrieved_docs, expected_sources):
        """Mean Reciprocal Rank - position of first relevant doc."""
        if not expected_sources:
            return 1.0

        expected_set = set(expected_sources)
        for rank, doc in enumerate(retrieved_docs, 1):
            if Path(doc.metadata.get("source", "")).name in expected_set:
                return 1.0 / rank

        return 0.0

    def precision_at_k(self, retrieved_docs, expected_sources, k=4):
        """Precision@K - fraction of top-k that are relevant."""
        if not expected_sources:
            return 1.0

        expected_set = set(expected_sources)
        relevant = sum(
            1 for doc in retrieved_docs[:k]
            if Path(doc.metadata.get("source", "")).name in expected_set
        )

        return relevant / k if k > 0 else 0.0

    def faithfulness(self, answer, retrieved_docs):
        """Faithfulness - overlap between answer and context."""
        if not answer or not retrieved_docs:
            return 0.0

        answer_words = set(answer.lower().split())
        context = " ".join([doc.page_content for doc in retrieved_docs]).lower()
        context_words = set(context.split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words & context_words)
        return min(overlap / len(answer_words), 1.0)

    def answer_relevance(self, answer, question):
        """Answer Relevance - semantic similarity to question."""
        if not answer or not question:
            return 0.0

        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([question, answer])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
            return float(sim[0][0])
        except:
            return 0.0

    def rouge_l(self, answer, ground_truth):
        """ROUGE-L F1 score."""
        if not answer or not ground_truth:
            return 0.0

        try:
            scores = self.rouge_scorer.score(ground_truth, answer)
            return scores['rougeL'].fmeasure
        except:
            return 0.0

    def bleu(self, answer, ground_truth):
        """BLEU score."""
        if not answer or not ground_truth:
            return 0.0

        try:
            reference = [ground_truth.split()]
            candidate = answer.split()
            return float(sentence_bleu(reference, candidate, smoothing_function=self.smoothing))
        except:
            return 0.0

    def cosine_similarity(self, answer, ground_truth):
        """Cosine similarity between answer and ground truth."""
        if not answer or not ground_truth:
            return 0.0

        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([ground_truth, answer])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
            return float(sim[0][0])
        except:
            return 0.0


class EvaluationEngine:
    """Runs the evaluation framework."""

    def __init__(self):
        self.config = EvaluationConfig()
        self.metrics = MetricsCalculator()
        self.results = {"summary": [], "detailed": {}}

    def run(self, corpus_docs, test_questions):
        """Run full evaluation."""
        print("\n" + "="*70)
        print("Running RAG Evaluation Framework")
        print("="*70)

        # Initialize embeddings
        print("\n  [1/4] Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs={"device": self.config.EMBEDDING_DEVICE}
        )

        # Initialize LLM
        print("  [2/4] Initializing LLM...")
        try:
            llm = OllamaLLM(
                model=self.config.LLM_MODEL
            )
            llm.invoke("test")
        except Exception as e:
            print(f"LLM Error: {e}")
            print("Make sure Ollama is running (default: http://localhost:11434)")
            return False

        # Test each strategy
        for strategy_name, cfg in self.config.CHUNK_STRATEGIES.items():
            print(f"\nStrategy: {strategy_name} (size={cfg['chunk_size']})")

            try:
                db_path = f"{self.config.EVAL_DB_DIR}_{strategy_name}"
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)

                # Split documents
                splitter = CharacterTextSplitter(
                    chunk_size=cfg['chunk_size'],
                    chunk_overlap=cfg['chunk_overlap'],
                    separator="\n"
                )
                chunks = splitter.split_documents(corpus_docs)

                # Create vector store
                vs = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=db_path
                )

                # Create QA chain
                retriever = vs.as_retriever(search_kwargs={"k": self.config.RETRIEVER_K})
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )

                # Evaluate on test questions
                results = []
                for q in tqdm(test_questions, desc=f"    {strategy_name}", leave=False):
                    try:
                        r = qa.invoke({"query": q["question"]})
                        answer = r["result"]
                        sources = r.get("source_documents", [])
                        expected = q.get("source_documents", [])

                        metric_result = {
                            "q_id": q["id"],
                            "hit_rate": self.metrics.hit_rate(sources, expected),
                            "mrr": self.metrics.mrr(sources, expected),
                            "p@4": self.metrics.precision_at_k(sources, expected),
                            "faithfulness": self.metrics.faithfulness(answer, sources),
                            "answer_rel": self.metrics.answer_relevance(answer, q["question"]),
                            "rouge_l": self.metrics.rouge_l(answer, q["ground_truth"]),
                            "bleu": self.metrics.bleu(answer, q["ground_truth"]),
                            "cosine": self.metrics.cosine_similarity(answer, q["ground_truth"]),
                        }
                        results.append(metric_result)
                    except:
                        pass

                df = pd.DataFrame(results)
                summary = {
                    "strategy": strategy_name,
                    "size": cfg["chunk_size"],
                    "avg_hit_rate": df["hit_rate"].mean(),
                    "avg_mrr": df["mrr"].mean(),
                    "avg_p@4": df["p@4"].mean(),
                    "avg_faithfulness": df["faithfulness"].mean(),
                    "avg_answer_rel": df["answer_rel"].mean(),
                    "avg_rouge_l": df["rouge_l"].mean(),
                    "avg_bleu": df["bleu"].mean(),
                    "avg_cosine": df["cosine"].mean(),
                }

                self.results["summary"].append(summary)
                self.results["detailed"][strategy_name] = results

                # Cleanup
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)

                print(f"Complete")

            except Exception as e:
                print(f"Failed: {e}")

        return True

    def save_results(self):
        """Save results to JSON."""
        with open(self.config.RESULTS_FILE, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {self.config.RESULTS_FILE}")

    def print_summary(self):
        """Print summary."""
        print("\n" + "="*70)
        print("Evaluation Summary")
        print("="*70 + "\n")

        if not self.results["summary"]:
            print("  No results")
            return

        df = pd.DataFrame(self.results["summary"]).set_index("strategy")

        print(df[["avg_hit_rate", "avg_mrr", "avg_p@4", "avg_faithfulness", 
                   "avg_rouge_l", "avg_bleu", "avg_cosine"]].to_string())

        print("\n" + "-"*70)
        print("Evaluation complete!")
        print("-"*70 + "\n")


def load_corpus(path):
    """Load all txt files from corpus folder."""
    try:
        loader = DirectoryLoader(
            path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")
        return docs
    except Exception as e:
        print(f"Failed: {e}")
        return None


def load_test_dataset(path):
    """Load test questions."""
    try:
        with open(path) as f:
            data = json.load(f)
        questions = data.get("test_questions", [])
        print(f"Loaded {len(questions)} test questions")
        return questions
    except Exception as e:
        print(f"Failed: {e}")
        return None


def main():
    print("\n" + "="*70)
    print("AmbedkarGPT: Evaluation Framework")
    print("="*70)

    print("\nLoading data...")
    corpus = load_corpus(EvaluationConfig.CORPUS_PATH)
    if not corpus:
        return

    questions = load_test_dataset(EvaluationConfig.TEST_DATASET_PATH)
    if not questions:
        return

    engine = EvaluationEngine()
    if engine.run(corpus, questions):
        engine.save_results()
        engine.print_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")