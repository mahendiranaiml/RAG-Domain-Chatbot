from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import traceback
from src.retriever import ChunkDataRetriever
from src.generator import generate_response

load_dotenv(".env")

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

data = {}

app = FastAPI(title="RAG Domain Chatbot")

@app.get("/")
def read_index():
    return FileResponse(INDEX_FILE)

@app.get("/health")
def health_check():
    return {"status": "ok"}

class ChatRequest(BaseModel):
    message: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        data.clear()

        from src.ingest import embed
        chunks, vectorstore_path = embed(str(file_path))
        
        data["chunks"] = chunks
        data["vectorstore_path"] = vectorstore_path
        data["retriever"] = ChunkDataRetriever(
            chunks=data["chunks"],
            vectorstore_path=data["vectorstore_path"]
        )

        pages_with_tables = sorted(
            {
                chunk.metadata.get("page_number")
                for chunk in chunks
                if chunk.metadata.get("has_tables")
            }
        )
        pages_with_images = sorted(
            {
                chunk.metadata.get("page_number")
                for chunk in chunks
                if chunk.metadata.get("has_images")
            }
        )

        return {
            "message": f"File uploaded completely! Processed {len(chunks)} chunks.",
            "num_chunks": len(chunks),
            "pages_with_tables": pages_with_tables,
            "pages_with_images": pages_with_images,
            "document_name": file.filename,
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)
        return JSONResponse(
            status_code=500,
            content={"message": f"Server Error: {str(e)}", "detail": error_details}
        )

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        question = request.message.strip()

        if not question:
            return JSONResponse(
                status_code=400,
                content={"message": "Question cannot be empty."}
            )

        if "chunks" not in data or "vectorstore_path" not in data:
            return JSONResponse(
                status_code=400,
                content={"message": "Please upload and process a PDF first."}
            )

        reranked_docs = data["retriever"].retriever(question)
        final_pages_for_llm = data["retriever"].final_pages_to_llm(reranked_docs)
        answer = generate_response(question, data["retriever"].llm, final_pages_for_llm)

        return {"answer": answer}

    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)
        return JSONResponse(
            status_code=500,
            content={"message": f"Server Error: {str(e)}", "detail": error_details}
        )
