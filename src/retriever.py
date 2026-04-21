from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
import os
import torch
import pickle
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from abc import ABC, abstractmethod
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import yaml
import logging
from dotenv import load_dotenv
load_dotenv() 

# Get GROQ API from .env
groq_api_key = os.getenv("GROQ")
if not groq_api_key:
    raise ValueError("Missing GROQ API key in environment.")

# Load Configure.yaml file's information
with open("configure.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Setup Logging
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s - %(module)s:%(lineno)d"
)
logger = logging.getLogger(__name__)



class ChunkRetriever(ABC):

    @abstractmethod
    def retriever(self, query : str):
        pass


class ChunkDataRetriever(ChunkRetriever):

    def __init__(self, chunks=None, vectorstore_path=None):
        if chunks is None:
            chunk_files = list(Path("cache/chunks").glob("*.pkl"))
            if not chunk_files:
                raise FileNotFoundError("No cached chunks found. Run DataEmbeddor first.")
            chunk_path = max(chunk_files, key=lambda path: path.stat().st_mtime)
            with chunk_path.open("rb") as ch:
                chunks = pickle.load(ch)

        if vectorstore_path is None:
            vectorstore_dir = Path("cache/vectorstore")
            vectorstore_paths = [path for path in vectorstore_dir.iterdir() if path.is_dir()] if vectorstore_dir.exists() else []
            if not vectorstore_paths:
                raise FileNotFoundError("No cached vectorstore found. Run DataEmbeddor first.")
            vectorstore_path = max(vectorstore_paths, key=lambda path: path.stat().st_mtime)

        self.chunks = chunks
        self.llm = ChatGroq(
            model = "llama-3.3-70b-versatile",
            api_key = groq_api_key,
            temperature = 0.0
        )


        model_path = "./embedding_model"
        
        # Load the custom downloaded embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_path if os.path.exists(model_path) else config["ingestor"]["embeddings"]["model_name"],
            model_kwargs = {"device":"cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # Load the local vectorstore that was created during ingestion
        self.vectorstore = FAISS.load_local(str(vectorstore_path), embedding_model, allow_dangerous_deserialization=True)

        # Pipe : Hybrid Retriever
        keyword_retriever = BM25Retriever.from_documents(self.chunks)
        keyword_retriever.k = config["retriever"]["keyword_retriever_chunks_size"]

        vector_retriever = self.vectorstore.as_retriever(search_kwargs = {"k":config["retriever"]["vector_retriever_chunks_size"]})

        self.hybrid_retriever = EnsembleRetriever(
            retrievers = [keyword_retriever, vector_retriever],
            weights = config["retriever"]["keyword_and_vector_retriever_weights"]
        )
        logger.info(f"Ensemble Retriever Created, weights of keyword and vector retriever are : {config['retriever']['keyword_and_vector_retriever_weights']}")


        prompt = PromptTemplate.from_template("""
            You are an AI assistant helping to retrieve relevant 
            information from a document.

            Your task is to generate 5 different search queries for the 
            following question to maximize document retrieval coverage.

            Follow these rules strictly:
            1. Break compound questions into simple sub-questions
            2. Use synonyms and alternative terminology
            3. Focus on ONE concept per query
            4. Keep each query short (2-5 words ideal)
            5. Vary vocabulary — don't repeat same words across queries

            Original Question: {question}

            Output ONLY the 5 queries, one per line, no numbering, 
            no explanations, no extra text:
        """)
        # Pipe : Muliti Query Generator with hybrid search
        advanced_retriever = MultiQueryRetriever.from_llm(
            retriever = self.hybrid_retriever,
            llm = self.llm,
            prompt = prompt
        )

        # Pipe : Final Docs after reranking
        compressor = FlashrankRerank(top_n = config["retriever"]["reranker_top_n"])
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=advanced_retriever
        )
        logger.info("Contextual Compression Retriever Created")

    def retriever(self, query):

        reranked_docs = self.compression_retriever.invoke(query)
        if self._needs_first_page_context(query):
            first_page_docs = [doc for doc in self.chunks if doc.metadata.get("page") == 0][:3]
            reranked_docs = self._merge_unique_docs(first_page_docs, reranked_docs)
        logger.info(f"Reranked Docs Created, size of retrieved docs : {len(reranked_docs)}")
        return reranked_docs

    # Pipe : After reranking full page of chunks must be provided to llm for final answer
    def final_pages_to_llm(self, reranked_docs):
        relevant_pages = {chunk.metadata.get("page") for chunk in reranked_docs}
        
        final_pages_for_llm = [chunk for chunk in self.chunks if chunk.metadata.get("page") in relevant_pages]
        
        logger.info(f"Retrieved {len(final_pages_for_llm)} chunks from {len(relevant_pages)} unique pages.")
        return final_pages_for_llm

    @staticmethod
    def _needs_first_page_context(query: str) -> bool:
        lowered = query.lower()
        keywords = ["title", "first page", "document name", "heading", "author", "paper name"]
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    def _merge_unique_docs(priority_docs, docs):
        merged = []
        seen = set()
        for doc in [*priority_docs, *docs]:
            key = (
                doc.metadata.get("page"),
                doc.metadata.get("chunk_type"),
                doc.page_content[:120],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return merged
