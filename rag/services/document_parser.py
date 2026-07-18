from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


class DocumentParserService:
    def parse(self, filename: str, file_bytes: bytes) -> tuple[str, str]:
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(file_bytes), "pypdf"

        if suffix == ".docx":
            return self._parse_docx(file_bytes), "python-docx"

        if suffix == ".txt":
            return file_bytes.decode("utf-8"), "plain-text"

        raise ValueError("Unsupported file type. Supported types: .pdf, .docx, .txt")

    def _parse_pdf(self, file_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    def _parse_docx(self, file_bytes: bytes) -> str:
        document = Document(BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
        return "\n".join(paragraphs).strip()
