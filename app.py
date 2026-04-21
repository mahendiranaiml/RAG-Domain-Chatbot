from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import asyncio
import traceback

load_dotenv(".env")

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

data = {}
app = FastAPI(title="RAG Domain Chatbot")


class ChatRequest(BaseModel):
    message: str


def load_retriever():
    from src.retriever import ChunkDataRetriever

    return ChunkDataRetriever()


@app.on_event("startup")
async def startup_event():
    try:
        data["retriever"] = await asyncio.to_thread(load_retriever)
    except Exception as e:
        data["startup_error"] = str(e)
        print(traceback.format_exc())


@app.get("/")
def read_index():
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health_check():
    return {"status": "ok"}


async def get_retriever():
    if "retriever" not in data:
        data["retriever"] = await asyncio.to_thread(load_retriever)
        data.pop("startup_error", None)
    return data["retriever"]


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        question = request.message.strip()

        if not question:
            return JSONResponse(status_code=400, content={"message": "Question cannot be empty."})

        retriever = await get_retriever()
        reranked_docs = await asyncio.to_thread(retriever.retriever, question)
        final_pages_for_llm = retriever.final_pages_to_llm(reranked_docs)

        from src.generator import generate_response

        answer = await asyncio.to_thread(
            generate_response,
            question,
            retriever.llm,
            final_pages_for_llm,
        )

        return {"answer": answer}
    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)
        return JSONResponse(
            status_code=500,
            content={"message": f"Server Error: {str(e)}", "detail": error_details},
        )
