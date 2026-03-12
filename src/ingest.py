from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from abc import abstractmethod, ABC
from typing import List
from pathlib import Path


from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HF_TOKEN")


class DataIngestor(ABC):

    """
    
    
    """
    @abstractmethod
    def doc_loader(self, filepath : str) -> List[Document]:
        pass

    @abstractmethod
    def chunker(self, document : List[Document]) -> List[Document]:
        pass

    @abstractmethod
    def embed_vectorstore(self, chunks : List[Document]) -> None:
        pass




class DataEmbeddor(DataIngestor):
    def __init__(self, filepath : str):
        self.filepath = filepath
 
    # Loads data from PDF
    def doc_loader(self) -> List[Document]:
        path = Path(self.filepath)
        loader = PyPDFLoader(path)
        docs = loader.load()

        return docs

    # Creates Chunk
    def chunker(self, document : List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
        separators = ["\n"],
        chunk_size = 200,
        chunk_overlap = 40
             )

        chunks = splitter.split_documents(document)

        return chunks




    # Embed Chunks into vectors then load in FAISS DB
    def embed_vectorstore(self, chunks : List[Document]) -> None:
        embeddings = HuggingFaceEmbeddings(

            model_name = "all-MiniLM-L6-v2"
        
        )
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        vectorstore.save_local("vectorstore")

# Pipe : for Loading data till Creating vector store
def embedder(filepath : str) -> None:
    obj = DataEmbeddor(filepath)

    docs = obj.doc_loader()
    chunks = obj.chunker(docs)
    obj.embed_vectorstore(chunks)











