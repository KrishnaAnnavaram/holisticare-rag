"""Knowledge base ingestion: source registry, document loading and section-aware chunks.

Every document must have an entry in ``sources.json`` in the document folder. The entry gives
the title, the license, the URL, the evidence label and the tradition of the document. A
document without an entry gets the evidence label ``unknown`` and a warning.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

EVIDENCE_LABELS = ("evidence_based", "traditional", "unknown")
TEXT_SUFFIXES = (".md", ".txt")
PDF_SUFFIX = ".pdf"


@dataclass(frozen=True)
class Source:
    file: str
    title: str
    license: str = "unknown"
    url: str = ""
    evidence: str = "unknown"
    tradition: str = "unknown"

    def __post_init__(self) -> None:
        if self.evidence not in EVIDENCE_LABELS:
            raise ValueError(f"{self.file}: evidence must be one of {EVIDENCE_LABELS}")


@dataclass
class Chunk:
    chunk_id: str
    source: str  # relative file path
    title: str
    section: str
    page: int | None
    text: str
    evidence: str
    tradition: str


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    warnings: list[str] = field(default_factory=list)


def load_registry(docs_dir: Path) -> dict[str, Source]:
    path = docs_dir / "sources.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for entry in data:
        s = Source(**entry)
        out[s.file.replace("\\", "/")] = s
    return out


def list_documents(docs_dir: Path) -> list[Path]:
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"{docs_dir} does not exist; run `holisticare synth` or see data/README.md")
    files = [p for p in sorted(docs_dir.rglob("*")) if p.suffix.lower() in (*TEXT_SUFFIXES, PDF_SUFFIX)]
    if not files:
        raise FileNotFoundError(f"no .md, .txt or .pdf files in {docs_dir}")
    return files


def docs_fingerprint(docs_dir: Path, settings_key: str) -> str:
    """SHA-256 of every document path and content, the registry and the chunk settings."""
    h = hashlib.sha256(settings_key.encode())
    reg = docs_dir / "sources.json"
    for p in [*list_documents(docs_dir), *([reg] if reg.exists() else [])]:
        h.update(p.relative_to(docs_dir).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == PDF_SUFFIX:
        try:
            from pypdf import PdfReader  # noqa: PLC0415  (optional extra)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError('PDF files need: pip install "holisticare-rag[pdf]"') from exc
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(None, path.read_text(encoding="utf-8", errors="replace"))]


def _sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown text on headings. Text before the first heading has the section name ``""``."""
    parts: list[tuple[str, str]] = []
    current, buf = "", []
    for line in text.splitlines():
        m = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", line)
        if m:
            if "".join(buf).strip():
                parts.append((current, "\n".join(buf).strip()))
            current, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        parts.append((current, "\n".join(buf).strip()))
    return parts


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split on paragraph and sentence borders. A piece is at most ``max_chars`` long."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    out, cur = [], ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        while len(p) > max_chars:  # a very long sentence: hard cut
            out.append(p[:max_chars])
            p = p[max_chars - overlap :]
        if cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur)
            cur = cur[-overlap:].lstrip() + " " + p if overlap else p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out


def ingest(docs_dir: Path, max_chars: int = 900, overlap: int = 120) -> tuple[list[Chunk], IngestReport]:
    docs_dir = Path(docs_dir)
    registry = load_registry(docs_dir)
    rep = IngestReport()
    if not registry:
        rep.warnings.append("no sources.json: every document has the evidence label `unknown`")
    chunks: list[Chunk] = []
    for path in list_documents(docs_dir):
        rel = path.relative_to(docs_dir).as_posix()
        src = registry.get(rel)
        if src is None:
            if registry:
                rep.warnings.append(f"{rel} is not in sources.json: evidence label `unknown`")
            src = Source(file=rel, title=path.stem.replace("_", " "))
        rep.documents += 1
        for page, text in _pages(path):
            for section, body in _sections(text):
                for i, piece in enumerate(split_text(body, max_chars, overlap)):
                    cid = hashlib.sha1(f"{rel}|{page}|{section}|{i}|{piece}".encode()).hexdigest()[:12]
                    chunks.append(Chunk(cid, rel, src.title, section, page, piece, src.evidence, src.tradition))
    rep.chunks = len(chunks)
    if not chunks:
        raise ValueError("the documents have no text")
    return chunks, rep


def chunk_to_dict(c: Chunk) -> dict:
    return asdict(c)
