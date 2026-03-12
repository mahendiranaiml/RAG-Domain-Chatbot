from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from abc import ABC, abstractmethod
from typing import List



class ChunkRetriever(ABC):

    @abstractmethod
    def retriever(self, query : str) -> List:
        pass




class ChunkDataRetriever(ChunkRetriever):

def __init__(self, docs, vectorstore):

    self.docs = docs
    self.vectorstore = vectorstore
    # Loading .env
    load_dotenv()

    # Getting Groq token
    groq_token = os.getenv("GROK")

    self.llm = ChatGroq(
        model = "llama-3.3-70b-versatile",
        temperature = 0.77,
        api_key = groq_token

    )

    # Pipe : Hybrid Retriever
    keyword_retriever = BM25Retriever.from_documents(self.docs)
    keyword_retriever.k = 3

    vector_retriever = self.vectorstore.as_retriever(search_kwargs = {"k":3})

    self.hybrid_retriever = EnsembleRetriever(
        retrievers = [keyword_retriever, vector_retriever],
        weights = [0.4, 0.6]
    )


    def retriever(self, query) -> List:


        # Pipe : Muliti Query Generator with hybrid search
        advanced_retriever = MultiQueryRetriever.from_llm(
            retriever = self.hybrid_retriever,
            llm = self.llm
        )

        # Pipe : Final Docs after reranking
        compressor = FlashrankRerank(top_n = 4)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=advanced_retriever
        )

        # Top 4 Chunks
        return compression_retriever.invoke(query)

def chunk_retriever(query : str, docs : List, vectorstore)-> List:
    obj = ChunkDataRetriever(docs, vectorstore)
    data = obj.retriever(query)
    return data
