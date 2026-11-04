"""Structure-aware recursive chunker.

Splits on the coarsest separator that keeps pieces under `chunk_size` characters
(sections → paragraphs → lines → sentences → words), then greedily packs pieces
back together with `overlap` characters of trailing context. Markdown headings are
tracked so every chunk carries its section path — which is fed into both the
lexical index and the prompt, a cheap but large win for retrieval precision.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

SEPARATORS = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    ordinal: int
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def indexed_text(self) -> str:
        return f"{self.section}\n{self.text}" if self.section else self.text


def _split(text: str, size: int, seps: tuple[str, ...]) -> list[str]:
    if len(text) <= size or not seps:
        if len(text) <= size:
            return [text]
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    if sep not in text:
        return _split(text, size, rest)
    parts = text.split(sep)
    pieces = [p + (sep if i < len(parts) - 1 else "") for i, p in enumerate(parts)]
    out: list[str] = []
    for p in pieces:
        out.extend([p] if len(p) <= size else _split(p, size, rest))
    return [p for p in out if p.strip()]


def _section_at(text: str, start: int, body: str) -> str:
    """Heading path in effect at the chunk's first line (headings opening the chunk count)."""
    offset = start + len(body.lstrip().split("\n", 1)[0]) + (len(body) - len(body.lstrip()))
    line_end = text.find("\n", offset)
    offset = len(text) if line_end < 0 else line_end
    path: list[tuple[int, str]] = []
    for m in _HEADING.finditer(text, 0, offset):
        level = len(m.group(1))
        path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, m.group(2).strip())]
    return " > ".join(t for _, t in path)


def chunk_document(
    text: str,
    doc_id: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    pieces = _split(text, chunk_size - overlap, SEPARATORS)
    chunks: list[Chunk] = []
    buf, buf_start, cursor = "", 0, 0
    for piece in pieces:
        pos = text.find(piece, cursor)
        pos = cursor if pos < 0 else pos
        if buf and len(buf) + len(piece) > chunk_size - overlap:
            section = _section_at(text, buf_start, buf)
            chunks.append(_make(buf, doc_id, len(chunks), section, metadata))
            tail = buf[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            buf, buf_start = tail, pos - len(tail)
        if not buf:
            buf_start = pos
        buf += piece
        cursor = pos + len(piece)
    if buf.strip():
        chunks.append(_make(buf, doc_id, len(chunks), _section_at(text, buf_start, buf), metadata))
    return chunks


def _make(text: str, doc_id: str, ordinal: int, section: str,
          metadata: dict[str, Any] | None) -> Chunk:
    body = text.strip()
    cid = hashlib.sha1(f"{doc_id}:{ordinal}:{body}".encode()).hexdigest()[:16]
    return Chunk(cid, doc_id, body, ordinal, section, dict(metadata or {}))
