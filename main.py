"""
AmbedkarGPT: Phase 1 - RAG Prototype
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

from langchain_classic.chains import RetrievalQA

# CONFIGURATION SECTION 

class Config:
    """Centralized configuration for the RAG system."""

    CHUNK_SIZE = 500 
    CHUNK_OVERLAP = 50

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE = "cpu"

    VECTOR_DB_PATH = "./chroma_db"
    INPUT_DATA_PATH = "./corpus"

    LLM_MODEL = "mistral:latest"
    LLM_TEMPERATURE = 0.1
    LLM_TOP_K = 10
    LLM_TOP_P = 0.5

    RETRIEVER_K = 6 

    DOCUMENT_LOADER_GLOB = "**/*.txt"

class AmbedkarGPT:
    """Implements the RAG pipeline for Q&A over Dr. Ambedkar's works."""

    def __init__(self, config: Config = Config()):
        """Initializes the RAG system with the given configuration."""
        self.config = config
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        print(" Initialized.")

    def _init_embeddings(self) -> bool:
        """Initializes the sentence-transformer embeddings."""
        try:
            print(f"  [1/4] Loading embeddings: {self.config.EMBEDDING_MODEL} (Device: {self.config.EMBEDDING_DEVICE})")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': self.config.EMBEDDING_DEVICE}
            )
            return True
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            print("     (Hint: Try `pip install -U sentence-transformers`)")
            return False

    def _init_vector_db(self) -> bool:
        """Loads or creates the Chroma vector database."""
        db_path = str(Path(self.config.VECTOR_DB_PATH).resolve())
        data_path = str(Path(self.config.INPUT_DATA_PATH).resolve())

        if not os.path.exists(data_path) or not os.listdir(data_path):
            print(f"Error: Input data directory is empty or missing.")
            print(f"     Please add.txt files to: {data_path}")
            return False

        try:
            if not os.path.exists(db_path):
                print(f"  [2/4] No existing DB found. Creating new vector store...")
                print(f"        - Loading documents from: {data_path}")
                
                # Use TextLoader for explicit.txt handling
                loader = TextLoader(
                    file_path=os.path.join(data_path, "ambedkar_corpus.txt"), 
                    encoding="utf-8"
                )
                documents = loader.load()

                if not documents:
                    print("Error: No documents loaded. Check glob pattern and data directory.")
                    return False
                
                print(f"Splitting {len(documents)} document(s) into chunks...")
                text_splitter = CharacterTextSplitter(
                    chunk_size=self.config.CHUNK_SIZE,
                    chunk_overlap=self.config.CHUNK_OVERLAP
                )
                texts = text_splitter.split_documents(documents)
                
                print(f"Found {len(texts)} text chunks.")
                print(f"Creating vector store at: {db_path}")
                
                self.vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=self.embeddings,
                    persist_directory=db_path
                )
                print("Vector store created and persisted.")
            
            else:
                print(f"Loading existing vector store from: {db_path}")
                self.vector_store = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embeddings
                )
                print("Vector store loaded.")

            return True

        except Exception as e:
            print(f"Error initializing vector DB: {e}")
            print("     (Hint: Check file permissions and `chromadb` installation.)")
            return False

    def _init_llm(self) -> bool:
        """Initializes the Ollama LLM."""
        try:
            print(f"  [3/4] Initializing LLM: {self.config.LLM_MODEL}")
            self.llm = OllamaLLM(
                model=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMPERATURE,
                top_k=self.config.LLM_TOP_K,
                top_p=self.config.LLM_TOP_P
            )
            # Test connection
            self.llm.invoke("Hi")
            print("LLM connection successful.")
            return True
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            print("     (Hint: Is Ollama running? Run `ollama serve` in a separate terminal.)")
            return False

    def _init_qa_chain(self) -> bool:
        """Initializes the RAG retrieval chain."""
        if not self.vector_store or not self.llm:
            print("  ❌ Cannot initialize QA chain: Vector store or LLM not ready.")
            return False
        
        try:
            print("Initializing RAG QA chain...")
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.RETRIEVER_K}
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            print("QA chain ready.")
            return True
        except Exception as e:
            print(f"Error initializing QA chain: {e}")
            return False

    def setup(self) -> bool:
        """Runs the complete setup pipeline."""
        print("\n" + "="*70)
        print("Initializing AmbedkarGPT Pipeline...")
        print("="*70)
        start_time = time.time()

        if (self._init_embeddings() and
            self._init_vector_db() and
            self._init_llm() and
            self._init_qa_chain()):
            
            end_time = time.time()
            print("="*70)
            print(f"Pipeline setup complete in {end_time - start_time:.2f} seconds.")
            print("="*70 + "\n")
            return True
        
        print("="*70)
        print("Pipeline setup failed.")
        print("="*70 + "\n")
        return False

    def ask(self, query: str) -> Tuple[Optional[str], Optional[list]]:
        """Asks a question to the RAG pipeline."""
        if not self.qa_chain:
            return "Error: QA chain is not initialized. Please run setup.", None
        
        try:
            response = self.qa_chain.invoke(query)
            return response.get('result'), response.get('source_documents')
        except Exception as e:
            print(f"Error during query: {e}")
            return f"Error: {e}", None

# INTERACTIVE SESSION

def run_interactive_session(rag: AmbedkarGPT):
    """Runs a command-line loop to chat with the RAG system."""
    print("Welcome to AmbedkarGPT!")
    print("Type your question about Dr. Ambedkar's works and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*70)

    question_count = 0
    while True:
        try:
            user_input = input("Q: ")
            
            if user_input.strip().lower() in ['quit', 'exit']:
                print(f"\n" + "*"*70)
                print(f"Thank you for using AmbedkarGPT. You asked {question_count} question(s).")
                print("*"*70)
                break

            if not user_input:
                print("(Please enter a question)")
                continue

            question_count += 1
            print("\n[Generating response...]")

            answer, sources = rag.ask(user_input)

            print(f"\nA: {answer}")

            if sources:
                print(f"\nSources:")
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get("source", "Unknown")
                    print(f"   [{i}] {source_name}")

            print("\n" + "-" * 70)

        except KeyboardInterrupt:
            print(f"\n\nSession interrupted. Answered {question_count} question(s).")
            break

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("(Please try another question)\n")


def main():
    """Main execution function."""
    try:
        rag = AmbedkarGPT()

        if not rag.setup():
            print("\nFailed to setup pipeline. Exiting.")
            sys.exit(1)

        run_interactive_session(rag)

    except KeyboardInterrupt:
        print("\n\nApplication interrupted.")
        sys.exit(0)

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()