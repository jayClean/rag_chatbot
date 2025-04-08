# loaders/pdf_loader.py
from io import BytesIO
import PyPDF2  # Using PyPDF2 instead of pypdf

def extract_pdf_chunks(filename: str, file_bytes: bytes):
    pdf_reader = PyPDF2.PdfFileReader(BytesIO(file_bytes))
    return [
        {"text": pdf_reader.getPage(p).extractText(), "metadata": {"source": filename, "page": p + 1}}
        for p in range(pdf_reader.getNumPages())
    ]
