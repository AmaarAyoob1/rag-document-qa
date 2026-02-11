"""
Document loader for PDF files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz


@dataclass
class DocumentPage:
    text: str
    page_number: int
    document_name: str
    document_path: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    name: str
    path: str
    pages: list[DocumentPage]
    total_pages: int

    @property
    def full_text(self) -> str:
        return chr(10).join(p.text for p in self.pages)


def load_pdf(file_path: str) -> Document:
    file_path = str(file_path)
    doc_name = Path(file_path).stem
    pdf = fitz.open(file_path)
    pages = []
    total = pdf.page_count
    for page_num in range(total):
        page = pdf[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue
        lines = [line.strip() for line in text.split(chr(10)) if line.strip()]
        cleaned_text = chr(10).join(lines)
        pages.append(DocumentPage(
            text=cleaned_text,
            page_number=page_num + 1,
            document_name=doc_name,
            document_path=file_path,
        ))
    pdf.close()
    return Document(name=doc_name, path=file_path, pages=pages, total_pages=total)


def load_documents(directory: str) -> list[Document]:
    documents = []
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    for pdf_file in sorted(dir_path.glob("*.pdf")):
        doc = load_pdf(str(pdf_file))
        documents.append(doc)
    return documents


def load_uploaded_pdf(file_bytes: bytes, filename: str) -> Document:
    doc_name = Path(filename).stem
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    total = pdf.page_count
    for page_num in range(total):
        page = pdf[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue
        lines = [line.strip() for line in text.split(chr(10)) if line.strip()]
        cleaned_text = chr(10).join(lines)
        pages.append(DocumentPage(
            text=cleaned_text,
            page_number=page_num + 1,
            document_name=doc_name,
            document_path=filename,
        ))
    pdf.close()
    return Document(name=doc_name, path=filename, pages=pages, total_pages=total)
