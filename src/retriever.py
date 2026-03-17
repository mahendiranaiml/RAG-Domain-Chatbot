from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from abc import ABC, abstractmethod
from typing import List



class ChunkRetriever(ABC):

    @abstractmethod
    def retriever(self, query : str):
        pass


class ChunkDataRetriever(ChunkRetriever):

    def __init__(self, chunks, vectorstore, llm):

        self.chunks = chunks
        self.vectorstore = vectorstore
        self.llm = llm

        # Pipe : Hybrid Retriever
        keyword_retriever = BM25Retriever.from_documents(self.chunks)
        keyword_retriever.k = 3

        vector_retriever = self.vectorstore.as_retriever(search_kwargs = {"k":3})

        self.hybrid_retriever = EnsembleRetriever(
            retrievers = [keyword_retriever, vector_retriever],
            weights = [0.5, 0.5]
        )


    def retriever(self, query):


        # Pipe : Muliti Query Generator with hybrid search
        advanced_retriever = MultiQueryRetriever.from_llm(
            retriever = self.hybrid_retriever,
            llm = self.llm
        )

        # Pipe : Final Docs after reranking
        compressor = FlashrankRerank(top_n = 3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=advanced_retriever
        )

        reranked_docs = compression_retriever.invoke(query)
        return reranked_docs

    # Pipe : After reranking full page of chunks must be provided to llm for final answer
    def final_pages_to_llm(self,reranked_docs, chunks):
        pages = []
        for chunk in reranked_docs:
            pages.append(chunk.metadata["page_label"])

        final_pages_for_llm = []
        for chunk in chunks:
            if (chunk.metadata["page_label"] in pages):
                final_pages_for_llm.append(chunk)
            
        # Top Relevant Pages
        return final_pages_for_llm

def chunk_retrieve(query : str, docs : List, vectorstore, chunks : List, llm):
    obj = ChunkDataRetriever(chunks, vectorstore, llm)
    reranked_docs = obj.retriever(query)
    final_pages_for_llm = obj.final_pages_to_llm(reranked_docs, chunks)
    return final_pages_for_llm
