import fitz


def extract_text_from_pdf(file_path: str):
    pages = []

    try:
        pdf_document = fitz.open(file_path)

        for page_index, page in enumerate(pdf_document):
            text = page.get_text()

            pages.append({
                "page_number": page_index + 1,
                "text": text.strip()
            })

        pdf_document.close()
        return pages

    except Exception as error:
        raise RuntimeError(f"PDF extraction failed: {str(error)}")