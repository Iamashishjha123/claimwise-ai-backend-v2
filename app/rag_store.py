import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRAGStore:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.documents = {}

    def add_document(self, document_id, file_name, chunks):
        texts = [chunk["text"] for chunk in chunks]

        if not texts:
            self.documents[document_id] = {
                "file_name": file_name,
                "chunks": [],
                "embeddings": None
            }
            return

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        self.documents[document_id] = {
            "file_name": file_name,
            "chunks": chunks,
            "embeddings": embeddings
        }

    def search(self, document_id, query, top_k=4):
        if document_id not in self.documents:
            return []

        doc = self.documents[document_id]

        if doc["embeddings"] is None or len(doc["chunks"]) == 0:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_embedding, doc["embeddings"])[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            chunk = doc["chunks"][idx]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "score": float(scores[idx]),
                "file_name": doc["file_name"]
            })

        return results


rag_store = SimpleRAGStore()