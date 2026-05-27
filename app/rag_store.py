
import re
from collections import Counter


def tokenize(text):
    text = text.lower()
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text)
    stopwords = {
        "the", "is", "are", "a", "an", "and", "or", "to", "of", "in", "for",
        "on", "with", "by", "as", "at", "from", "this", "that", "it", "be",
        "can", "will", "shall", "may", "your", "you", "we", "our"
    }
    return [word for word in words if word not in stopwords and len(word) > 2]


class SimpleRAGStore:
    """
    Lightweight in-memory retrieval store.
    Uses keyword overlap scoring instead of heavy embeddings.
    Works on Render free 512MB RAM.
    """

    def __init__(self):
        self.documents = {}

    def add_document(self, document_id, file_name, chunks):
        processed_chunks = []

        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            token_counts = Counter(tokens)

            processed_chunks.append({
                **chunk,
                "tokens": tokens,
                "token_counts": token_counts
            })

        self.documents[document_id] = {
            "file_name": file_name,
            "chunks": processed_chunks
        }

    def search(self, document_id, query, top_k=4):
        if document_id not in self.documents:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        results = []

        for chunk in self.documents[document_id]["chunks"]:
            chunk_counts = chunk["token_counts"]

            overlap_score = 0

            for token, count in query_counts.items():
                if token in chunk_counts:
                    overlap_score += min(count, chunk_counts[token])

            # Normalize score
            score = overlap_score / max(len(query_tokens), 1)

            if score > 0:
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "score": float(score),
                    "file_name": self.documents[document_id]["file_name"]
                })

        results.sort(key=lambda item: item["score"], reverse=True)

        return results[:top_k]


rag_store = SimpleRAGStore()
