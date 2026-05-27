def chunk_text_pages(pages, chunk_size=900, overlap=150):
    chunks = []

    for page in pages:
        page_number = page.get("page_number")
        text = page.get("text", "")

        if not text.strip():
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"page_{page_number}_chunk_{len(chunks) + 1}",
                    "page_number": page_number,
                    "text": chunk_text
                })

            start = end - overlap

            if start < 0:
                start = 0

            if start >= len(text):
                break

    return chunks