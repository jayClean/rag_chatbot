# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_pipeline import ingest_pdf, ingest_website, ask_llm_with_context
from embedder import embed_text
from retriever import search_index
from vector_store import is_index_empty, get_vector_count
import logging
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow frontend (localhost:5173) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or ["*"] for all origins (not recommended for prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WebsiteIngest(BaseModel):
    url: str

class QuestionInput(BaseModel):
    question: str

@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        chunks = ingest_pdf(file.filename, contents)
        return {"message": "PDF ingested", "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error ingesting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting PDF: {str(e)}")

@app.post("/ingest/website")
async def ingest_website_endpoint(request: WebsiteIngest):
    try:
        chunks = ingest_website(request.url)
        return {"message": "Website ingested", "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error ingesting website: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting website: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionInput):
    try:
        if not request.question:
            raise ValueError("Question cannot be empty")
            
        query_embedding = embed_text(request.question)
        chunks = search_index(query_embedding)
        
        if not chunks:
            return JSONResponse(content={"answer": "I don't have enough information to answer that question."}, status_code=200)
            
        answer = ask_llm_with_context(request.question, chunks)
        return {"answer": answer}
    except ValueError as e:
        logger.error(f"Value error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/status/vector-store")
def check_vector_store_status():
    try:
        return {
            "vector_store_empty": is_index_empty(),
            "vector_count": get_vector_count()
        }
    except Exception as e:
        logger.error(f"Error checking vector store status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking vector store status: {str(e)}")