from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import DistanceStrategy
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from abc import abstractmethod, ABC
from typing import List
from pathlib import Path
from hashlib import sha256
import time
import yaml
from pypdf import PdfReader

with open("configure.yaml", "r") as f:
    config = yaml.safe_load(f)

import logging 
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s - %(module)s:%(lineno)d"
)
logger = logging.getLogger(__name__)
PIPELINE_VERSION = "v2"


def _build_vectorstore_path(filepath: str) -> Path:
    source_path = Path(filepath).resolve()
    stat = source_path.stat()
    cache_key = f"{PIPELINE_VERSION}|{source_path}|{stat.st_size}|{stat.st_mtime_ns}"
    cache_name = sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return Path("vectorstore") / cache_name


class DataIngestor(ABC):

    """
    
    
    """
    @abstractmethod
    def doc_loader(self) -> List[Document]:
        pass

    @abstractmethod
    def chunker(self, document : List[Document]) -> List[Document]:
        pass

    @abstractmethod
    def embed_vectorstore(self, chunks : List[Document]) -> None:
        pass


class PDFParser:
   
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def load(self) -> List[Document]:
        if self._has_pymupdf():
            return self._load_with_pymupdf()
        return self._load_with_pypdf()

    def _load_with_pymupdf(self) -> List[Document]:
        import fitz

        documents: List[Document] = []
        pdf = fitz.open(self.filepath)

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            text = self._extract_page_text(page)
            table_blocks = self._extract_table_blocks(page, page_number)
            image_blocks = self._extract_fitz_image_blocks(pdf, page, page_number)

            if not text:
                scanned_text = self._ocr_scanned_page(page_number)
                if scanned_text:
                    text = scanned_text

            page_metadata = self._build_page_metadata(page_index, page_number, table_blocks, image_blocks)
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={**page_metadata, "chunk_type": "text"},
                    )
                )

            for table_index, table_block in enumerate(table_blocks, start=1):
                documents.append(
                    Document(
                        page_content=table_block,
                        metadata={**page_metadata, "chunk_type": "table", "table_index": table_index},
                    )
                )

            for image_index, image_block in enumerate(image_blocks, start=1):
                documents.append(
                    Document(
                        page_content=image_block,
                        metadata={**page_metadata, "chunk_type": "image", "image_index": image_index},
                    )
                )

            if not any([text, table_blocks, image_blocks]):
                documents.append(
                    Document(
                        page_content=f"[Page {page_number} contains no extractable text.]",
                        metadata={**page_metadata, "chunk_type": "empty"},
                    )
                )

        logger.info(f"Parsed {len(documents)} docs from {self.filepath} using PyMuPDF")
        return documents

    def _load_with_pypdf(self) -> List[Document]:
        reader = PdfReader(str(self.filepath))
        documents: List[Document] = []

        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            text = self._clean_text(page.extract_text(extraction_mode="layout") or "")
            image_blocks = self._extract_pypdf_image_blocks(page, page_number)

            if not text:
                scanned_text = self._ocr_scanned_page(page_number)
                if scanned_text:
                    text = scanned_text

            page_metadata = self._build_page_metadata(page_index, page_number, [], image_blocks)
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={**page_metadata, "chunk_type": "text"},
                    )
                )

            for image_index, image_block in enumerate(image_blocks, start=1):
                documents.append(
                    Document(
                        page_content=image_block,
                        metadata={**page_metadata, "chunk_type": "image", "image_index": image_index},
                    )
                )

            if not any([text, image_blocks]):
                documents.append(
                    Document(
                        page_content=f"[Page {page_number} contains no extractable text.]",
                        metadata={**page_metadata, "chunk_type": "empty"},
                    )
                )

        logger.info(f"Parsed {len(documents)} docs from {self.filepath} using pypdf fallback")
        return documents

    def _extract_page_text(self, page) -> str:
        blocks = page.get_text("blocks")
        text_parts = []

        for block in blocks:
            block_text = self._clean_text(block[4] if len(block) > 4 else "")
            if block_text:
                text_parts.append(block_text)

        return "\n\n".join(text_parts).strip()

    def _extract_table_blocks(self, page, page_number: int) -> List[str]:
        table_blocks: List[str] = []

        try:
            tables = page.find_tables()
        except Exception as exc:
            logger.warning(f"Table detection failed on page {page_number}: {exc}")
            return table_blocks

        for table_index, table in enumerate(getattr(tables, "tables", []), start=1):
            try:
                rows = table.extract()
            except Exception as exc:
                logger.warning(f"Table extraction failed on page {page_number} table {table_index}: {exc}")
                continue

            if not rows:
                continue

            header = [self._clean_cell(cell) for cell in rows[0]]
            row_lines = [f"Columns: {' | '.join(header)}"] if any(header) else []

            for row_number, row in enumerate(rows[1:], start=1):
                values = [self._clean_cell(cell) for cell in row]
                if any(values):
                    row_lines.append(f"Row {row_number}: {' | '.join(values)}")

            if row_lines:
                table_blocks.append(f"[Table {table_index} on page {page_number}]\n" + "\n".join(row_lines))

        return table_blocks

    def _extract_fitz_image_blocks(self, pdf, page, page_number: int) -> List[str]:
        image_blocks: List[str] = []

        for image_index, image in enumerate(page.get_images(full=True), start=1):
            xref = image[0]
            try:
                image_info = pdf.extract_image(xref)
                image_bytes = image_info.get("image", b"")
            except Exception as exc:
                logger.warning(f"Image extraction failed on page {page_number} image {image_index}: {exc}")
                continue

            image_text = self._ocr_image_bytes(image_bytes)
            label = f"[Image {image_index} on page {page_number}]"
            image_blocks.append(f"{label}\n{image_text or 'Embedded image detected.'}")

        return image_blocks

    def _extract_pypdf_image_blocks(self, page, page_number: int) -> List[str]:
        image_blocks: List[str] = []

        for image_index, image_file in enumerate(getattr(page, "images", []), start=1):
            image_text = self._ocr_image_bytes(image_file.data)
            label = f"[Image {image_index} on page {page_number}]"
            image_blocks.append(f"{label}\n{image_text or 'Embedded image detected.'}")

        return image_blocks

    def _ocr_image_bytes(self, image_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return ""

        try:
            image = Image.open(BytesIO(image_bytes))
            return self._clean_text(pytesseract.image_to_string(image))
        except Exception as exc:
            logger.warning(f"Image OCR failed: {exc}")
            return ""

    def _ocr_scanned_page(self, page_number: int) -> str:
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError:
            return ""

        try:
            pages = convert_from_path(
                str(self.filepath),
                first_page=page_number,
                last_page=page_number,
                fmt="png",
            )
            if not pages:
                return ""
            return self._clean_text(pytesseract.image_to_string(pages[0]))
        except Exception as exc:
            logger.warning(f"Scanned-page OCR failed for page {page_number}: {exc}")
            return ""

    def _build_page_metadata(
        self,
        page_index: int,
        page_number: int,
        table_blocks: List[str],
        image_blocks: List[str],
    ) -> dict:
        return {
            "source": str(self.filepath),
            "page": page_index,
            "page_number": page_number,
            "has_tables": bool(table_blocks),
            "has_images": bool(image_blocks),
            "table_count": len(table_blocks),
            "image_count": len(image_blocks),
        }

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

    @staticmethod
    def _clean_cell(value) -> str:
        return str(value or "").replace("\n", " ").strip()

    @staticmethod
    def _has_pymupdf() -> bool:
        try:
            import fitz  # noqa: F401
            return True
        except ImportError:
            return False




class DataEmbeddor(DataIngestor):
    def __init__(self, filepath : str):
        self.filepath = filepath
        self.vectorstore_path = _build_vectorstore_path(filepath)
 
    # Loads data from PDF
    def doc_loader(self) -> List[Document]:
        path = Path(self.filepath)

        if not path.exists():
            logger.critical(f"Data Ingestion Failed: Path {self.filepath} does not exist.")
            raise FileNotFoundError(f"Incorrect Path : {self.filepath}")

        if path.suffix.lower() != ".pdf":
            logger.critical(f"Data Ingestion Failed: File {self.filepath} is not a PDF file.")
            raise ValueError(f"File has no extension .pdf. PyPDFLoader only supports .pdf files.")

        if path.stat().st_size == 0:
            logger.critical(f"Data Ingestion Failed: File {self.filepath} is empty.")
            raise ValueError(f"File is Empty : {self.filepath}")

        docs = PDFParser(path).load()
        logger.info(f"Loaded docs : {len(docs)} from {self.filepath}")

        return docs
    
    # Creates Chunk
    def chunker(self, document : List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
        separators = config["ingestor"]["chunks"]["separators"],
        chunk_size = config["ingestor"]["chunks"]["chunk_size"],
        chunk_overlap = config["ingestor"]["chunks"]["chunk_overlap"]
        )

        chunks: List[Document] = []
        for doc in document:
            if doc.metadata.get("chunk_type") == "text":
                chunks.extend(splitter.split_documents([doc]))
            else:
                chunks.append(doc)
        logger.info(f"Created Chunks : {len(chunks)}")

        return chunks




    # Embed Chunks into vectors then load in FAISS DB
    def embed_vectorstore(self, chunks : List[Document]) -> Path:


        model_path = "./embedding_model"
        if not os.path.exists(model_path):
            model = SentenceTransformer(config["ingestor"]["embeddings"]["model_name"]) # Will Load Model from the path if exists otherwise download from internet
            model.save(model_path)

        embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs = {"device":"cuda" if __import__("torch").cuda.is_available() else "cpu"}
        )

        logger.info(f"Embeddings Model Loaded, Model Path : {model_path}")

        if self.vectorstore_path.exists():
            logger.info(f"Using cached vectorstore at {self.vectorstore_path}")
            return self.vectorstore_path

        start_time = time.time()
        vectorstore = FAISS.from_documents(
            chunks,
            embedding_model,
            distance_strategy= getattr(DistanceStrategy, config["ingestor"]["embeddings"]["distance_strategy"])
        )
        end_time = time.time()
        time_taken = end_time - start_time

        logger.info(f"Vector Store created and Time taken is : {time_taken}")
        self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.vectorstore_path))
        logger.info(f"Vectorstore Saved to {self.vectorstore_path}")
        return self.vectorstore_path


# Orchestrator : for Loading data till Creating vector store
def embed(filepath : str):
    obj = DataEmbeddor(filepath)

    docs = obj.doc_loader()
    chunks = obj.chunker(docs)
    vectorstore_path = obj.embed_vectorstore(chunks)
    
    return chunks, str(vectorstore_path)







