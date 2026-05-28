# ClaimWise AI

ClaimWise AI is a document-grounded insurance policy assistant. Users can upload insurance policy PDFs and ask questions about coverage, premiums, death benefits, cancellation, conversion privileges, and exclusions.

The system extracts text from PDFs, chunks the document, retrieves relevant sections, and returns answers with citations instead of hallucinating.

## Features

- PDF upload
- PDF text extraction
- Document chunking
- Lightweight RAG retrieval
- Question answering over uploaded documents
- Citations with page numbers
- Evidence preview
- Not-found guardrails for unsupported topics
- Frontend connected with backend API
- Deployed backend on Render

## Tech Stack

- FastAPI
- PyMuPDF
- Python
- Lightweight retrieval logic
- Render
- Emergent frontend

## Why this project matters

Insurance documents are hard to understand. Normal chatbots can hallucinate, which is risky in insurance. ClaimWise AI only answers from uploaded documents and shows evidence/citations.
