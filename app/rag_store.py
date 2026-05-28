
import re
from collections import Counter


STOPWORDS = {
    "the", "is", "are", "a", "an", "and", "or", "to", "of", "in", "for",
    "on", "with", "by", "as", "at", "from", "this", "that", "it", "be",
    "can", "will", "shall", "may", "your", "you", "we", "our", "under",
    "policy", "document", "uploaded", "about", "what", "when", "where",
    "how", "does", "do", "tell", "me" "insurance", "coverage", "benefits", "company", "life"
}


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    text = text.lower()
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text)
    return [word for word in words if word not in STOPWORDS and len(word) > 2]


def looks_like_table_of_contents(text):
    lower = text.lower()
    toc_signals = [
        "table of contents",
        "policy schedule",
        "terms used in this policy",
        "benefits provided by this policy",
    ]

    # Many short lines + section labels usually means TOC.
    has_toc_signal = any(signal in lower for signal in toc_signals)
    many_section_refs = len(re.findall(r"\b[A-F]\s*\d\b", text)) >= 4

    return has_toc_signal or many_section_refs


class SimpleRAGStore:
    """
    Lightweight in-memory retrieval store.
    Uses keyword scoring with boosts and TOC penalty.
    Designed for Render free tier.
    """

    def __init__(self):
        self.documents = {}

    def add_document(self, document_id, file_name, chunks):
        processed_chunks = []

        for chunk in chunks:
            text = clean_text(chunk["text"])
            tokens = tokenize(text)
            token_counts = Counter(tokens)

            processed_chunks.append({
                **chunk,
                "text": text,
                "tokens": tokens,
                "token_counts": token_counts,
                "is_toc": looks_like_table_of_contents(text)
            })

        self.documents[document_id] = {
            "file_name": file_name,
            "chunks": processed_chunks
        }

    def search(self, document_id, query, top_k=4):
        if document_id not in self.documents:
            return []

        query_clean = clean_text(query.lower())
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        results = []

        for chunk in self.documents[document_id]["chunks"]:
            text_lower = chunk["text"].lower()
            chunk_counts = chunk["token_counts"]

            score = 0.0

            # Basic keyword overlap
            for token, count in query_counts.items():
                if token in chunk_counts:
                    score += min(count, chunk_counts[token])

            # Boost if all important query tokens appear
            matched_tokens = [token for token in query_tokens if token in chunk_counts]
            match_ratio = len(set(matched_tokens)) / max(len(set(query_tokens)), 1)
            score += match_ratio * 2.0

            # Phrase boosts for insurance questions
            if "grace period" in query_clean and "grace period" in text_lower:
                score += 5.0

            if "premium" in query_clean and "premium" in text_lower:
                score += 2.0

            if "death benefit" in query_clean and "death benefit" in text_lower:
                score += 5.0

            if "conversion" in query_clean and "conversion privilege" in text_lower:
                score += 5.0

            if "exchange" in query_clean and "exchange privilege" in text_lower:
                score += 5.0

            if "beneficiary" in query_clean and "beneficiary" in text_lower:
                score += 4.0

            if "cancel" in query_clean and "cancel" in text_lower:
                score += 4.0

            # Penalize table of contents
            if chunk["is_toc"]:
                score *= 0.25

            # Prefer chunks with explanatory sentences, not index-only chunks
            if len(chunk["text"]) > 300 and "." in chunk["text"]:
                score += 1.0

            if score > 0:
                normalized_score = score / 10.0

                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "score": float(round(normalized_score, 4)),
                    "file_name": self.documents[document_id]["file_name"]
                })

        results.sort(key=lambda item: item["score"], reverse=True)

        return results[:top_k]


rag_store = SimpleRAGStore()