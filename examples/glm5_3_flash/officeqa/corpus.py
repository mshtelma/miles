"""Treasury Bulletin corpus and the four retrieval tools of the OfficeQA agent.

Documents are read from the pinned corpus zip, held as one immutable object per process. A tool result
may be 32,000 characters, so a 150K-token episode can hold whole tables instead of 40-line slices.

Search uses BM25 (``bm25s``, installed in the AIR image) when it is importable, otherwise keyword
overlap. A policy-supplied grep regex runs in the calling thread; a pathological pattern is bounded
only by the agent's tool deadline.
"""

import json
import re
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import bm25s
except ImportError:  # optional: search falls back to keyword overlap
    bm25s = None

CORPUS_ZIP = "treasury_bulletins_clean.zip"
CHUNKS_FILE = "chunks.jsonl"
SEARCH_TOP_K = 6
TOOL_MAX_CHARS = 32_000
READ_MAX_LINES = 400
SNIPPET_CHARS = 500
GREP_MAX_MATCHES = 40


@dataclass(frozen=True)
class Corpus:
    documents: dict[str, list[str]]  # bulletin file name -> lines
    chunks: list[dict[str, Any]]  # search passages with source and bulletin_date
    retriever: Any  # bm25s.BM25 over the chunks, or None for the keyword fallback


_load_lock = threading.Lock()
_loaded: dict[Path, Corpus] = {}


def load_corpus(data_dir: Path) -> Corpus:
    """Read the staged corpus once per process; concurrent first calls wait for the one load."""
    with _load_lock:
        if data_dir not in _loaded:
            _loaded[data_dir] = _read_corpus(data_dir)
        return _loaded[data_dir]


def search_documents(corpus: Corpus, query: str, top_k: int = SEARCH_TOP_K) -> str:
    q = str(query).strip()
    if len(q) < 2:
        return "Error: query must be at least 2 characters"
    k = max(1, min(_coerce_int(top_k, SEARCH_TOP_K), 20))
    ranked = _bm25_ranked(corpus, q, k) if corpus.retriever is not None else _keyword_ranked(corpus, q, k)
    if not ranked:
        return f"No passages found for {query!r}. Try different terms or a specific year."
    out = []
    for i, (score, chunk) in enumerate(ranked):
        source = chunk.get("source", "?")
        date = chunk.get("bulletin_date", "")
        out.append(
            f"[{i}] {source}{f' ({date})' if date else ''}  score={score:.2f}\n"
            f"    {_highlight(chunk.get('content', ''), q)}"
        )
    return _clip("\n\n".join(out))


def grep_documents(corpus: Corpus, pattern: str, file_name: str = "", year: str = "", regex: bool = False) -> str:
    pat = str(pattern)
    if len(pat.strip()) < 2:
        return "Error: pattern must be at least 2 characters"
    use_regex = bool(regex) and str(regex).lower() not in ("false", "0", "no", "")
    if use_regex:
        try:
            compiled = re.compile(pat, re.IGNORECASE)
        except re.error as error:
            return f"Error: invalid regex {pat!r}: {error}"

        def match(line: str) -> bool:
            return compiled.search(line) is not None

    else:
        needle = pat.lower()

        def match(line: str) -> bool:
            return needle in line.lower()

    files = [_norm_name(file_name)] if file_name else _corpus_files(corpus, year)
    if not files:
        return f"No documents to search{f' for year {year}' if year else ''}."
    hits, scanned = _grep_hits(corpus, files, match)
    if not hits:
        where = file_name or (f"year {year}" if year else "the corpus")
        return f"No matches for {pattern!r} in {where}."
    head = (
        f"{len(hits)} match(es) for {pattern!r}"
        f"{' (regex)' if use_regex else ''} in "
        f"{file_name or (f'{scanned} files for {year}' if year else f'{scanned} files')}:"
    )
    return _clip(head + "\n" + "\n".join(hits))


def read_document(corpus: Corpus, file_name: str, start_line: int = 0, num_lines: int = 100) -> str:
    name = _norm_name(file_name)
    lines = corpus.documents.get(name)
    if lines is None:
        return f"Error: no document named {name!r}. Use list_documents(year) or search_documents first."
    start = max(0, _coerce_int(start_line, 0))
    count = max(1, min(_coerce_int(num_lines, 100), READ_MAX_LINES))
    chunk = lines[start : start + count]
    if not chunk:
        return f"{name}: start_line {start} is past end ({len(lines)} lines)."
    body = "\n".join(f"{start + i}: {line}" for i, line in enumerate(chunk))
    return _clip(f"{name} lines {start}-{start + len(chunk) - 1} of {len(lines)}:\n{body}")


def list_documents(corpus: Corpus, year: str = "") -> str:
    names = _corpus_files(corpus, year)
    y = str(year).strip()
    if not names:
        return f"No documents found{f' for year {y}' if y else ''}."
    return _clip(f"{len(names)} document(s){f' for {y}' if y else ''}:\n" + ", ".join(names[:250]))


def _read_corpus(data_dir: Path) -> Corpus:
    with zipfile.ZipFile(data_dir / CORPUS_ZIP) as archive:
        documents = {
            Path(member).name: archive.read(member).decode("utf-8", errors="ignore").splitlines()
            for member in archive.namelist()
            if member.endswith(".txt")
        }
    with (data_dir / CHUNKS_FILE).open(encoding="utf-8", errors="ignore") as stream:
        chunks = [json.loads(line) for line in stream if line.strip()]
    return Corpus(documents=documents, chunks=chunks, retriever=_bm25_index(chunks))


def _bm25_index(chunks: list[dict[str, Any]]):
    if bm25s is None or not chunks:
        return None
    retriever = bm25s.BM25()
    contents = [chunk.get("content", "") for chunk in chunks]
    retriever.index(bm25s.tokenize(contents, show_progress=False), show_progress=False)
    return retriever


def _bm25_ranked(corpus: Corpus, query: str, k: int) -> list[tuple[float, dict]]:
    indices, scores = corpus.retriever.retrieve(
        bm25s.tokenize([query], show_progress=False), k=min(k, len(corpus.chunks)), show_progress=False
    )
    return [
        (float(score), corpus.chunks[int(index)])
        for index, score in zip(indices[0], scores[0], strict=True)
        if score > 0
    ]


def _keyword_ranked(corpus: Corpus, query: str, k: int) -> list[tuple[float, dict]]:
    terms = {term for term in query.lower().split() if len(term) >= 3}
    if not terms:
        return []
    ranked = []
    for chunk in corpus.chunks:
        text = chunk.get("content", "").lower()
        matched = sum(1 for term in terms if term in text)
        if matched:
            ranked.append((matched / len(terms), chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:k]


def _grep_hits(corpus: Corpus, files: list[str], match) -> tuple[list[str], int]:
    hits, scanned = [], 0
    for name in files:
        lines = corpus.documents.get(name)
        if not lines:
            continue
        scanned += 1
        for i, line in enumerate(lines):
            if match(line):
                hits.append(f"{name}:{i}: {line.strip()}")
                if len(hits) >= GREP_MAX_MATCHES:
                    return hits, scanned
    return hits, scanned


def _corpus_files(corpus: Corpus, year: str = "") -> list[str]:
    names = sorted(corpus.documents)
    y = str(year).strip()
    return [name for name in names if f"_{y}_" in name] if y else names


def _highlight(content: str, query: str) -> str:
    lowered = content.lower()
    pos = -1
    for term in (term for term in query.lower().split() if len(term) >= 3):
        found = lowered.find(term)
        if found != -1 and (pos == -1 or found < pos):
            pos = found
    if pos == -1:
        return content[:SNIPPET_CHARS] + ("..." if len(content) > SNIPPET_CHARS else "")
    start = max(0, pos - SNIPPET_CHARS // 3)
    end = min(len(content), pos + (2 * SNIPPET_CHARS) // 3)
    return ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")


def _clip(text: str) -> str:
    if len(text) <= TOOL_MAX_CHARS:
        return text
    return text[:TOOL_MAX_CHARS] + "\n...[truncated; refine the query / read a slice]"


def _coerce_int(value, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _norm_name(file_name: str) -> str:
    name = Path(str(file_name).strip()).name  # basename only: no traversal
    if name and not name.endswith(".txt"):
        name += ".txt"
    return name
