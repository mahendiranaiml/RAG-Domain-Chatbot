from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from abc import abstractmethod, ABC
from typing import List
from pathlib import Path
from hashlib import sha256
from pypdf import PdfReader
import os
import time
import yaml
import pickle
import logging


with open("configure.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s - %(module)s:%(lineno)d"
)
logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v3"
CHUNK_CACHE_DIR = Path("cache/chunks")
VECTORSTORE_DIR = Path("cache/vectorstore")


def _compute_file_hash(filepath: str | Path, buffer_size: int = 1024 * 1024) -> str:

    filepath = Path(filepath)
    hasher = sha256()

    with filepath.open("rb") as f:
        while chunk := f.read(buffer_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def _get_chunking_signature() -> str:
    chunk_cfg = config["ingestor"]["chunks"]
    signature_raw = (
        f"{PIPELINE_VERSION}|"
        f"{chunk_cfg['chunk_size']}|"
        f"{chunk_cfg['chunk_overlap']}|"
        f"{chunk_cfg['separators']}"
    )
    return sha256(signature_raw.encode("utf-8")).hexdigest()[:16]


def _build_chunk_cache_path(filepath: str | Path) -> Path:
    filepath = Path(filepath).resolve()
    file_hash = _compute_file_hash(filepath)
    chunk_signature = _get_chunking_signature()

    cache_key = f"{filepath}|{file_hash}|{chunk_signature}"
    cache_name = sha256(cache_key.encode("utf-8")).hexdigest()[:16]

    return CHUNK_CACHE_DIR / f"{cache_name}.pkl"


def _build_vectorstore_path(filepath: str | Path) -> Path:

    filepath = Path(filepath).resolve()
    file_hash = _compute_file_hash(filepath)
    chunk_signature = _get_chunking_signature()

    embedding_cfg = config["ingestor"]["embeddings"]
    vector_signature_raw = (
        f"{PIPELINE_VERSION}|"
        f"{file_hash}|"
        f"{chunk_signature}|"
        f"{embedding_cfg['model_name']}|"
        f"{embedding_cfg['distance_strategy']}"
    )
    cache_name = sha256(vector_signature_raw.encode("utf-8")).hexdigest()[:16]

    return VECTORSTORE_DIR / cache_name


class DataIngestor(ABC):

    @abstractmethod
    def doc_loader(self) -> List[Document]:
        pass

    @abstractmethod
    def chunker(self, document: List[Document]) -> List[Document]:
        pass

    @abstractmethod
    def embed_vectorstore(self, chunks: List[Document]) -> Path:
        pass


class PDFParser:

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def load(self) -> List[Document]:
        reader = PdfReader(str(self.filepath))
        documents: List[Document] = []

        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            text = self._extract_text(page)
            metadata = {
                "source": str(self.filepath),
                "page": page_index,
                "page_number": page_number,
                "has_tables": False,
                "has_images": False,
                "table_count": 0,
                "image_count": 0,
            }

            documents.append(
                Document(
                    page_content=text or f"[Page {page_number} contains no extractable text.]",
                    metadata={**metadata, "chunk_type": "text" if text else "empty"},
                )
            )

        logger.info(f"Parsed {len(documents)} docs from {self.filepath} using pypdf")
        return documents

    def _extract_text(self, page) -> str:
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            text = page.extract_text() or ""
        return self._clean_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        cleaned_lines: List[str] = []
        previous_blank = False

        for line in lines:
            is_blank = not line
            if is_blank and previous_blank:
                continue
            cleaned_lines.append(line)
            previous_blank = is_blank

        return "\n".join(cleaned_lines).strip()


class DataEmbeddor(DataIngestor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.chunk_cache_path = _build_chunk_cache_path(filepath)
        self.vectorstore_path = _build_vectorstore_path(filepath)

    def doc_loader(self) -> List[Document]:
        path = Path(self.filepath)

        if not path.exists():
            logger.critical(f"Data Ingestion Failed: Path {self.filepath} does not exist.")
            raise FileNotFoundError(f"Incorrect Path: {self.filepath}")

        if path.suffix.lower() != ".pdf":
            logger.critical(f"Data Ingestion Failed: File {self.filepath} is not a PDF file.")
            raise ValueError(f"File has no extension .pdf. Only PDF files are supported.")

        if path.stat().st_size == 0:
            logger.critical(f"Data Ingestion Failed: File {self.filepath} is empty.")
            raise ValueError(f"File is empty: {self.filepath}")

        docs = PDFParser(path).load()
        logger.info(f"Loaded docs: {len(docs)} from {self.filepath}")
        return docs

    def chunker(self, document: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            separators=config["ingestor"]["chunks"]["separators"],
            chunk_size=config["ingestor"]["chunks"]["chunk_size"],
            chunk_overlap=config["ingestor"]["chunks"]["chunk_overlap"]
        )

        chunks: List[Document] = []
        for doc in document:
            if doc.metadata.get("chunk_type") == "text":
                chunks.extend(splitter.split_documents([doc]))
            else:
                chunks.append(doc)

        logger.info(f"Created chunks: {len(chunks)}")
        return chunks

    def save_chunks(self, chunks: List[Document]) -> None:
        self.chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.chunk_cache_path.open("wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Chunk cache saved at {self.chunk_cache_path}")

    def load_chunks(self) -> List[Document]:
        with self.chunk_cache_path.open("rb") as f:
            chunks = pickle.load(f)
        logger.info(f"Loaded cached chunks from {self.chunk_cache_path}")
        return chunks

    def get_or_create_chunks(self) -> List[Document]:

        if self.chunk_cache_path.exists():
            return self.load_chunks()

        docs = self.doc_loader()
        chunks = self.chunker(docs)
        self.save_chunks(chunks)
        return chunks

    def embed_vectorstore(self, chunks: List[Document]) -> Path:
        model_path = "./embedding_model"

        if not os.path.exists(model_path):
            model = SentenceTransformer(config["ingestor"]["embeddings"]["model_name"])
            model.save(model_path)

        embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda" if __import__("torch").cuda.is_available() else "cpu"}
        )

        logger.info(f"Embedding model loaded from: {model_path}")

        if self.vectorstore_path.exists():
            logger.info(f"Using cached vectorstore at {self.vectorstore_path}")
            return self.vectorstore_path

        start_time = time.time()
        vectorstore = FAISS.from_documents(
            chunks,
            embedding_model,
            distance_strategy=getattr(
                DistanceStrategy,
                config["ingestor"]["embeddings"]["distance_strategy"]
            )
        )
        end_time = time.time()

        logger.info(f"Vector store created in {end_time - start_time:.2f} seconds")

        self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.vectorstore_path))
        logger.info(f"Vectorstore saved at {self.vectorstore_path}")
        return self.vectorstore_path

    def run(self) -> tuple[List[Document], Path]:
        chunks = self.get_or_create_chunks()
        vectorstore_path = self.embed_vectorstore(chunks)
        return chunks, vectorstore_path


def embed(filepath: str | Path) -> tuple[List[Document], Path]:
    return DataEmbeddor(filepath).run()


if __name__ == "__main__":
    file_path = "uploads/NIPS-2017-attention-is-all-you-need-Paper.pdf"

    try:
        chunks, vectorstore_path = DataEmbeddor(file_path).run()
        print(f"Chunks ready: {len(chunks)}")
        print(f"Vectorstore stored at: {vectorstore_path}")

    except Exception as e:
        print(f"Error occurred: {e}")
