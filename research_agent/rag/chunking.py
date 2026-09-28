"""Structure-aware chunk packing for Markdown-like academic documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


FALLBACK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""]
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_TABLE_SEPARATOR = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")


@dataclass(frozen=True)
class StructuredUnit:
    text: str
    kind: str
    section_title: str = ""


def parse_structured_units(text: str, initial_section: str = "") -> list[StructuredUnit]:
    """Parse headings, tables, code blocks and paragraphs into indivisible units."""
    lines = str(text or "").splitlines()
    units: list[StructuredUnit] = []
    paragraph: list[str] = []
    section = initial_section

    def flush_paragraph() -> None:
        if not paragraph:
            return
        value = "\n".join(paragraph).strip()
        paragraph.clear()
        if value:
            units.append(StructuredUnit(value, "paragraph", section))

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        heading = _HEADING.match(stripped)
        if heading:
            flush_paragraph()
            section = heading.group(2).strip()
            units.append(StructuredUnit(stripped, "heading", section))
            index += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            code = [line]
            index += 1
            while index < len(lines):
                code.append(lines[index].rstrip())
                if lines[index].strip().startswith("```"):
                    index += 1
                    break
                index += 1
            units.append(StructuredUnit("\n".join(code), "code", section))
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and _TABLE_SEPARATOR.match(
            lines[index + 1].strip()
        ):
            flush_paragraph()
            table = [line, lines[index + 1].rstrip()]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append(lines[index].rstrip())
                index += 1
            units.append(StructuredUnit("\n".join(table), "table", section))
            continue
        if not stripped:
            flush_paragraph()
        else:
            paragraph.append(line)
        index += 1
    flush_paragraph()
    return units


def _split_table(unit: StructuredUnit, chunk_size: int) -> list[StructuredUnit]:
    lines = unit.text.splitlines()
    if len(lines) < 3 or len(unit.text) <= chunk_size:
        return [unit]
    header = lines[:2]
    chunks: list[StructuredUnit] = []
    current = header[:]
    for row in lines[2:]:
        candidate = "\n".join([*current, row])
        if len(candidate) > chunk_size and len(current) > 2:
            chunks.append(StructuredUnit("\n".join(current), "table", unit.section_title))
            current = [*header, row]
        else:
            current.append(row)
    if len(current) > 2:
        chunks.append(StructuredUnit("\n".join(current), "table", unit.section_title))
    return chunks or [unit]


def _expand_oversized_units(
    units: Iterable[StructuredUnit], chunk_size: int
) -> list[StructuredUnit]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=FALLBACK_SEPARATORS,
    )
    expanded: list[StructuredUnit] = []
    for unit in units:
        if len(unit.text) <= chunk_size:
            expanded.append(unit)
        elif unit.kind == "table":
            expanded.extend(_split_table(unit, chunk_size))
        elif unit.kind == "code" and len(unit.text) <= chunk_size * 2:
            # Slightly oversized code/formula blocks are safer intact than split
            # in the middle of a statement.
            expanded.append(unit)
        else:
            expanded.extend(
                StructuredUnit(part, unit.kind, unit.section_title)
                for part in splitter.split_text(unit.text)
                if part.strip()
            )
    return expanded


def split_structured_document(
    document: Document,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Pack complete structural units while preserving bounded unit overlap."""
    if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("invalid structured chunk size/overlap")
    initial_section = str(document.metadata.get("section_title") or "")
    units = _expand_oversized_units(
        parse_structured_units(document.page_content, initial_section), chunk_size
    )
    if not units:
        return []

    chunks: list[Document] = []
    current: list[StructuredUnit] = []

    def emit() -> None:
        if not current:
            return
        text = "\n\n".join(unit.text for unit in current).strip()
        if not text:
            return
        metadata = dict(document.metadata)
        sections = [unit.section_title for unit in current if unit.section_title]
        if sections:
            metadata["section_title"] = sections[-1]
        metadata["content_types"] = ",".join(sorted({unit.kind for unit in current}))
        metadata["structure_chunk_index"] = len(chunks)
        chunks.append(Document(page_content=text, metadata=metadata))

    for unit in units:
        candidate_length = sum(len(item.text) for item in current) + len(unit.text)
        candidate_length += max(0, len(current)) * 2
        if current and candidate_length > chunk_size:
            emit()
            overlap_units: list[StructuredUnit] = []
            overlap_length = 0
            for previous in reversed(current):
                if previous.kind in {"table", "code"}:
                    continue
                additional = len(previous.text) + (2 if overlap_units else 0)
                if overlap_length + additional > chunk_overlap:
                    break
                overlap_units.insert(0, previous)
                overlap_length += additional
            current = overlap_units
        current.append(unit)
    emit()
    return chunks

