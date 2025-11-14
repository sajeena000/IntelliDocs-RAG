import io
from typing import List
from pypdf import PdfReader


def extract_text_from_pdf(data: io.BytesIO) -> str:
    reader = PdfReader(data)
    pages = []
    for page in reader.pages:
        try:
            raw_text = page.extract_text() or ""
            cleaned_text = raw_text.replace('\x00', '')
            pages.append(cleaned_text)
        except Exception:
            continue
    return "\n".join(pages)


def extract_text_from_txt(data: io.BytesIO) -> str:
    content = data.read()
    try:
        decoded_text = content.decode("utf-8", errors="ignore")
    except Exception:
        decoded_text = content.decode("latin-1", errors="ignore")
    
    return decoded_text.replace('\x00', '')