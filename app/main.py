import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pdf_parser import extract_text_from_pdf
from app.chunker import chunk_text_pages
from app.rag_store import rag_store


app = FastAPI(
    title="ClaimWise AI Backend",
    description="FastAPI backend for health insurance claim document processing and lightweight RAG retrieval",
    version="1.0.0"
)

FRONTEND_URL = "https://health-claims-ai-1.preview.emergentagent.com"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class PageResponse(BaseModel):
    page_number: int
    text: str


class UploadResponse(BaseModel):
    status: str
    message: str
    document_id: str
    file_name: str
    total_pages: int
    total_chunks: int
    preview_text: str
    pages: List[PageResponse]
    uploaded_at: str


class AskRequest(BaseModel):
    document_id: str
    question: str


class CitationResponse(BaseModel):
    file_name: str
    page_number: int
    score: float


class RetrievedContextResponse(BaseModel):
    chunk_id: str
    page_number: int
    text: str
    score: float
    file_name: str


class AskResponse(BaseModel):
    status: str
    answer: str
    confidence: str
    citations: List[CitationResponse]
    retrieved_context: List[RetrievedContextResponse]
    best_score: Optional[float] = None
    evidence_preview: Optional[str] = None


@app.get("/")
def home():
    return {
        "message": "ClaimWise AI FastAPI backend is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ClaimWise AI Backend",
        "time": datetime.utcnow().isoformat()
    }


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    safe_filename = file.filename.replace(" ", "_")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    stored_filename = f"{timestamp}_{safe_filename}"
    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pages = extract_text_from_pdf(file_path)

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from this PDF"
            )

        total_pages = len(pages)

        preview_text = ""
        for page in pages:
            if page.get("text"):
                preview_text = page["text"][:1200]
                break

        if not preview_text:
            raise HTTPException(
                status_code=400,
                detail="This PDF does not contain readable text. It may be scanned or image-based."
            )

        document_id = str(uuid.uuid4())
        chunks = chunk_text_pages(pages)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Text was extracted, but no valid chunks could be created."
            )

        rag_store.add_document(
            document_id=document_id,
            file_name=file.filename,
            chunks=chunks
        )

        return {
            "status": "success",
            "message": "PDF uploaded, text extracted, and indexed successfully",
            "document_id": document_id,
            "file_name": file.filename,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "preview_text": preview_text,
            "pages": pages[:3],
            "uploaded_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong while processing the PDF: {str(error)}"
        )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):

    if not request.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    results = rag_store.search(
        document_id=request.document_id,
        query=request.question,
        top_k=4
    )

    if not results:
        return {
            "status": "not_found",
            "answer": (
                "I could not find relevant information in the uploaded document. "
                "Try asking with words that appear in the document, such as premium, death benefit, grace period, conversion, exchange, beneficiary, or cancellation."
            ),
            "confidence": "low",
            "citations": [],
            "retrieved_context": [],
            "best_score": None,
            "evidence_preview": None
        }

    best_score = results[0]["score"]

    citations = [
        {
            "file_name": item["file_name"],
            "page_number": item["page_number"],
            "score": item["score"]
        }
        for item in results
    ]

    evidence_text = "\n\n".join(
        [
            f"Page {item['page_number']}: {item['text'][:700]}"
            for item in results
        ]
    )

    if best_score < 0.10:
        return {
            "status": "low_confidence",
            "answer": (
                "I found some related text, but the evidence is weak. "
                "Please review the cited evidence or ask a more specific question using wording from the document."
            ),
            "confidence": "low",
            "citations": citations,
            "retrieved_context": results,
            "best_score": best_score,
            "evidence_preview": evidence_text
        }

    confidence = "high" if best_score >= 0.45 else "medium"

    top_evidence = results[0]["text"][:900]

    clean_answer = (
        "Based on the uploaded document, the most relevant section says:\n\n"
        f"{top_evidence}\n\n"
        "Please review the cited page for full context."
    )

    return {
        "status": "success",
        "answer": clean_answer,
        "confidence": confidence,
        "citations": citations,
        "retrieved_context": results,
        "best_score": best_score,
        "evidence_preview": evidence_text
    }