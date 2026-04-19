# RAG Domain Chatbot

Single-document PDF chatbot built for text-heavy business and report PDFs.

## What it supports

- one PDF uploaded at a time
- page-aware text extraction
- table detection and extraction
- embedded image detection
- optional OCR for scanned pages and image text
- hybrid retrieval with reranking

## Best test document

- Unilever Annual Report and Accounts 2024

## Setup

Create and activate your environment, then install dependencies:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

Optional OCR support:

```bash
pip install pillow pytesseract pdf2image
```

You also need these system tools if you want OCR:

- Tesseract OCR
- Poppler

## Environment

Create `.env` with:

```env
GROQ=your_groq_api_key
```

## Run

```bash
uvicorn app:app --reload --port 8001
```

Open:

- `http://127.0.0.1:8001`
- `http://127.0.0.1:8001/health`

## What happens on upload

1. the PDF is saved in `uploads/`
2. the parser reads the file page by page
3. text becomes text chunks
4. tables become separate table chunks
5. images become separate image chunks
6. chunks are embedded and stored in `vectorstore/`
7. the app shows which pages contain tables and images

## Recommended questions

- What is the exact title of the document?
- Which pages contain tables?
- What are the column names in Table 1?
- Read row 2 of Table 1 exactly.
- Which pages contain images?
- Is there any text extracted from images?
- Summarize the first page.

## Current architecture

- `app.py`: FastAPI app and upload/chat endpoints
- `src/pdf_pipeline.py`: PDF parsing for text, tables, and images
- `src/ingest.py`: chunking and embeddings
- `src/retriever.py`: hybrid retrieval and reranking
- `src/generator.py`: grounded answer generation

## Notes

- PyMuPDF is the preferred parser.
- If PyMuPDF is unavailable, the app falls back to pypdf.
- OCR is optional and used only when dependencies are installed.
