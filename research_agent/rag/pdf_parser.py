"""Layout-aware parser for academic PDF documents.

The parser uses text coordinates and font metadata rather than flattening a
page immediately. It reconstructs a deterministic reading order for common
single/two-column papers, removes repeated margin text, preserves headings and
tables as Markdown, and records page-level diagnostics. OCR is opt-in because
it requires a server-side Tesseract installation and can distort formulas.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from langchain_core.documents import Document

from research_agent.config import (
    get_pdf_extract_tables,
    get_pdf_max_file_mb,
    get_pdf_max_pages,
    get_pdf_min_text_chars,
    get_pdf_ocr_language,
    get_pdf_ocr_mode,
)
from research_agent.errors import DocumentParseError


PARSER_NAME = "academic_pdf_pymupdf_v1"
_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+\S|"
    r"abstract|introduction|background|related work|method(?:s|ology)?|"
    r"experiments?|results?|discussion|limitations?|conclusions?|"
    r"references|acknowledg(?:e)?ments?|appendix)(?:\s|$)",
    re.IGNORECASE,
)
_CAPTION_PATTERN = re.compile(r"^(?:fig(?:ure)?|table)\s*[.:]?\s*\d+", re.IGNORECASE)
_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_MATH_CHARS = set("=±×÷∑∏√∫≈≠≤≥∞∂∇αβγδεζηθλμπρστφψω")


@dataclass(frozen=True)
class PDFBlock:
    page_number: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float = 0.0
    bold: bool = False
    kind: str = "paragraph"

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def area(self) -> float:
        return self.width * max(0.0, self.y1 - self.y0)


def normalize_margin_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    normalized = re.sub(r"\d+", "#", normalized)
    return normalized.strip("-–—|· ")


def repeated_margin_signatures(
    pages: Iterable[tuple[float, float, list[PDFBlock]]]
) -> set[str]:
    """Find short header/footer strings repeated on multiple pages."""
    pages = list(pages)
    if len(pages) < 2:
        return set()
    counts: Counter[str] = Counter()
    for _, page_height, blocks in pages:
        page_signatures: set[str] = set()
        for block in blocks:
            in_margin = block.y1 <= page_height * 0.12 or block.y0 >= page_height * 0.88
            if not in_margin or len(block.text) > 180 or block.text.count("\n") > 1:
                continue
            signature = normalize_margin_signature(block.text)
            if signature:
                page_signatures.add(signature)
        counts.update(page_signatures)
    threshold = max(2, math.ceil(len(pages) * 0.4))
    return {signature for signature, count in counts.items() if count >= threshold}


def _is_repeated_margin(
    block: PDFBlock,
    page_height: float,
    signatures: set[str],
) -> bool:
    in_margin = block.y1 <= page_height * 0.12 or block.y0 >= page_height * 0.88
    return in_margin and normalize_margin_signature(block.text) in signatures


def order_page_blocks(
    blocks: Iterable[PDFBlock], page_width: float, page_height: float
) -> list[PDFBlock]:
    """Return a stable reading order for common one/two-column paper layouts."""
    values = [block for block in blocks if block.text.strip()]
    if not values:
        return []
    spanning = [
        block for block in values
        if block.width >= page_width * 0.58
        and block.x0 <= page_width * 0.28
        and block.x1 >= page_width * 0.72
    ]
    narrow = [block for block in values if block not in spanning]
    left = [block for block in narrow if (block.x0 + block.x1) / 2 < page_width * 0.48]
    right = [block for block in narrow if (block.x0 + block.x1) / 2 > page_width * 0.52]
    vertically_related = any(
        min(left_block.y1, right_block.y1)
        >= max(left_block.y0, right_block.y0) - page_height * 0.06
        for left_block in left
        for right_block in right
    )
    two_columns = bool(left and right and vertically_related)
    if not two_columns:
        return sorted(values, key=lambda block: (round(block.y0, 1), block.x0, block.kind))

    spanning.sort(key=lambda block: (block.y0, block.x0))
    regions: dict[int, list[PDFBlock]] = {index: [] for index in range(len(spanning) + 1)}
    for block in narrow:
        region = sum(1 for full in spanning if full.y1 <= block.y0 + 1.0)
        regions[region].append(block)

    ordered: list[PDFBlock] = []
    for region in range(len(spanning) + 1):
        segment = regions[region]
        segment.sort(key=lambda block: (
            0 if (block.x0 + block.x1) / 2 < page_width / 2 else 1,
            block.y0,
            block.x0,
        ))
        ordered.extend(segment)
        if region < len(spanning):
            ordered.append(spanning[region])
    return ordered


def join_pdf_lines(lines: Iterable[str]) -> str:
    """Join visual lines while repairing common end-of-line hyphenation."""
    result = ""
    for raw in lines:
        line = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not line:
            continue
        if not result:
            result = line
        elif result.endswith("-") and line[:1].islower():
            result = result[:-1] + line
        elif line[0] in ",.;:!?)]}，。；：！？、":
            result += line
        else:
            result += " " + line
    return result.strip()


def markdown_table(rows: Iterable[Iterable[Any]]) -> str:
    normalized: list[list[str]] = []
    for row in rows:
        values = [
            re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", "\\|")
            for cell in row
        ]
        if any(values):
            normalized.append(values)
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    if not any(header):
        header = [f"column_{index}" for index in range(1, width + 1)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _intersection_ratio(
    bbox: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> float:
    x0 = max(bbox[0], container[0])
    y0 = max(bbox[1], container[1])
    x1 = min(bbox[2], container[2])
    y1 = min(bbox[3], container[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return intersection / area


def _text_blocks(page: Any, page_number: int, textpage: Any = None) -> list[PDFBlock]:
    kwargs = {"textpage": textpage} if textpage is not None else {}
    payload = page.get_text("dict", sort=False, **kwargs)
    blocks: list[PDFBlock] = []
    for raw_block in payload.get("blocks", []):
        if raw_block.get("type") != 0:
            continue
        lines: list[str] = []
        sizes: list[float] = []
        bold = False
        for raw_line in raw_block.get("lines", []):
            line_parts: list[str] = []
            for span in raw_line.get("spans", []):
                text = str(span.get("text") or "")
                if text.strip():
                    line_parts.append(text)
                    sizes.append(float(span.get("size") or 0.0))
                    font = str(span.get("font") or "").lower()
                    flags = int(span.get("flags") or 0)
                    bold = bold or "bold" in font or bool(flags & 16)
            line = "".join(line_parts).strip()
            if line:
                lines.append(line)
        text = join_pdf_lines(lines)
        bbox = tuple(float(value) for value in raw_block.get("bbox", (0, 0, 0, 0)))
        if text and len(bbox) == 4:
            blocks.append(PDFBlock(
                page_number=page_number,
                bbox=bbox,
                text=text,
                font_size=max(sizes) if sizes else 0.0,
                bold=bold,
            ))
    return blocks


def _table_blocks(page: Any, page_number: int) -> list[PDFBlock]:
    if not get_pdf_extract_tables() or not hasattr(page, "find_tables"):
        return []
    try:
        finder = page.find_tables()
    except Exception:
        return []
    blocks: list[PDFBlock] = []
    for table in getattr(finder, "tables", []) or []:
        try:
            text = markdown_table(table.extract())
            bbox = tuple(float(value) for value in table.bbox)
        except Exception:
            continue
        if text and len(bbox) == 4:
            blocks.append(PDFBlock(
                page_number=page_number,
                bbox=bbox,
                text=text,
                kind="table",
            ))
    return blocks


def classify_pdf_blocks(blocks: Iterable[PDFBlock]) -> list[PDFBlock]:
    values = list(blocks)
    body_candidates = [
        block.font_size for block in values
        if block.kind == "paragraph" and len(block.text) >= 80 and block.font_size > 0
    ]
    fallback_sizes = [block.font_size for block in values if block.font_size > 0]
    body_size = median(body_candidates or fallback_sizes or [10.0])
    result: list[PDFBlock] = []
    for block in values:
        if block.kind == "table":
            result.append(block)
            continue
        stripped = block.text.strip()
        short = len(stripped) <= 180
        heading_pattern = short and bool(_HEADING_PATTERN.match(stripped))
        visual_heading = short and block.font_size >= body_size * 1.18
        bold_heading = short and block.bold and block.font_size >= body_size * 1.02
        if heading_pattern or visual_heading or bold_heading:
            if block.font_size >= body_size * 1.55:
                prefix = "#"
            elif block.font_size >= body_size * 1.25:
                prefix = "##"
            else:
                prefix = "###"
            result.append(replace(block, text=f"{prefix} {stripped}", kind="heading"))
            continue
        if _CAPTION_PATTERN.match(stripped):
            result.append(replace(block, kind="caption"))
            continue
        math_ratio = sum(character in _MATH_CHARS for character in stripped) / max(
            len(stripped), 1
        )
        if len(stripped) <= 300 and math_ratio >= 0.08:
            result.append(replace(block, kind="equation"))
            continue
        result.append(block)
    return result


def extract_page_blocks(
    page: Any,
    page_number: int,
    *,
    textpage: Any = None,
    include_tables: bool = True,
) -> list[PDFBlock]:
    text_blocks = _text_blocks(page, page_number, textpage=textpage)
    tables = _table_blocks(page, page_number) if include_tables else []
    if tables:
        text_blocks = [
            block for block in text_blocks
            if not any(_intersection_ratio(block.bbox, table.bbox) >= 0.55 for table in tables)
        ]
    return classify_pdf_blocks([*text_blocks, *tables])


class AcademicPDFLoader:
    """Load an academic PDF into page-level structured Markdown Documents."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[Document]:
        try:
            file_size = self.path.stat().st_size
        except OSError as exc:
            raise DocumentParseError(f"Cannot read PDF: {self.path}") from exc
        max_bytes = get_pdf_max_file_mb() * 1024 * 1024
        if file_size > max_bytes:
            raise DocumentParseError(
                f"PDF exceeds PDF_MAX_FILE_MB: {self.path.name}"
            )
        try:
            import fitz
        except ImportError as exc:
            raise DocumentParseError(
                "PyMuPDF is required for academic PDF parsing."
            ) from exc
        try:
            pdf = fitz.open(str(self.path))
        except Exception as exc:
            raise DocumentParseError(f"Cannot open PDF: {self.path.name}") from exc
        if pdf.needs_pass:
            pdf.close()
            raise DocumentParseError(f"Encrypted PDF requires a password: {self.path.name}")
        if pdf.page_count > get_pdf_max_pages():
            pdf.close()
            raise DocumentParseError(
                f"PDF exceeds PDF_MAX_PAGES: {self.path.name}"
            )

        ocr_mode = get_pdf_ocr_mode()
        ocr_language = get_pdf_ocr_language()
        minimum_chars = get_pdf_min_text_chars()
        raw_pages: list[dict[str, Any]] = []
        try:
            for page_index, page in enumerate(pdf):
                blocks = extract_page_blocks(page, page_index)
                extracted_chars = sum(
                    len(re.sub(r"\s+", "", block.text)) for block in blocks
                )
                warnings: list[str] = []
                ocr_used = False
                if extracted_chars < minimum_chars:
                    if ocr_mode == "tesseract":
                        try:
                            textpage = page.get_textpage_ocr(
                                language=ocr_language,
                                dpi=200,
                                full=True,
                            )
                            blocks = extract_page_blocks(
                                page,
                                page_index,
                                textpage=textpage,
                                include_tables=False,
                            )
                            extracted_chars = sum(
                                len(re.sub(r"\s+", "", block.text))
                                for block in blocks
                            )
                            ocr_used = True
                            warnings.append("ocr_layout_is_approximate")
                            if extracted_chars < minimum_chars:
                                warnings.append("ocr_low_text_density")
                        except Exception as exc:
                            raise DocumentParseError(
                                "Tesseract OCR was requested but failed for "
                                f"{self.path.name} page {page_index + 1}: {exc}"
                            ) from exc
                    else:
                        warnings.append("low_text_density_possible_scan")
                raw_pages.append({
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "blocks": blocks,
                    "warnings": warnings,
                    "ocr_used": ocr_used,
                    "extracted_chars": extracted_chars,
                })
        finally:
            metadata = dict(pdf.metadata or {})
            page_count = pdf.page_count
            pdf.close()

        repeated = repeated_margin_signatures([
            (item["width"], item["height"], item["blocks"])
            for item in raw_pages
        ])
        # A completely scanned page may produce no Document and would otherwise
        # lose its warning.  Propagate paper-level page diagnostics to every
        # surviving page so retrieval and the UI can expose incomplete parsing.
        document_warnings = {
            f"page_{page_index + 1}:{warning}"
            for page_index, item in enumerate(raw_pages)
            for warning in item["warnings"]
        }
        documents: list[Document] = []
        current_section = ""
        doi = ""
        for page_index, item in enumerate(raw_pages):
            blocks = [
                block for block in item["blocks"]
                if not _is_repeated_margin(block, item["height"], repeated)
            ]
            ordered = order_page_blocks(blocks, item["width"], item["height"])
            page_sections: list[str] = []
            for block in ordered:
                if block.kind == "heading":
                    current_section = block.text.lstrip("# ").strip()
                    page_sections.append(current_section)
            content = "\n\n".join(block.text for block in ordered if block.text.strip())
            if not doi:
                match = _DOI_PATTERN.search(content)
                doi = match.group(0).rstrip(".,;)") if match else ""
            if not content.strip():
                continue
            warnings = [*item["warnings"], *document_warnings]
            if repeated:
                warnings.append("repeated_headers_footers_removed")
            content_types = sorted({block.kind for block in ordered})
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": str(self.path),
                    "file_name": self.path.name,
                    "file_path": str(self.path),
                    "page": page_index,
                    "page_number": page_index + 1,
                    "page_start": page_index + 1,
                    "page_end": page_index + 1,
                    "page_count": page_count,
                    "section_title": page_sections[0] if page_sections else current_section,
                    "section_titles": " | ".join(page_sections),
                    "content_types": ",".join(content_types),
                    "parser": PARSER_NAME,
                    "parser_warnings": ",".join(sorted(set(warnings))),
                    "ocr_used": bool(item["ocr_used"]),
                    "doi": doi,
                    "pdf_title": str(metadata.get("title") or ""),
                    "pdf_author": str(metadata.get("author") or ""),
                },
            ))
        if not documents:
            hint = (
                "Enable PDF_OCR_MODE=tesseract and install Tesseract on the server."
                if ocr_mode == "disabled"
                else "Check the PDF and installed OCR language data."
            )
            raise DocumentParseError(
                f"No extractable PDF text in {self.path.name}. {hint}"
            )
        return documents

