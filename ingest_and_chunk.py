#!/usr/bin/env python3
"""Ingest local and URL documents, save raw text, clean text, and generate overlapping chunks.

Default settings follow planning.md:
- chunk size: 900 chars (target range 800-1000)
- overlap: 180 chars (target range 150-200)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".html", ".htm", ".json", ".csv", ".pdf"}
JSONDict = dict[str, Any]

BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"(?i)^\s*(skip to (main )?content|sign in|log in|log out|menu|navigation)\s*$"),
    re.compile(r"(?i)^\s*(cookie(?:s)?|privacy policy|terms of service)\s*$"),
    re.compile(r"(?i)^\s*(read more|show more|load more|share|subscribe|follow|back to top)\s*$"),
    re.compile(r"(?i)^\s*(comment count|top comments?|comments?)\s*[:\-()]?\s*$"),
    re.compile(r"(?i)^\s*section titled\b.*$"),
    re.compile(r"(?i)^\s*on this page\b.*$"),
    re.compile(r"(?i)^\s*copy page(?: as markdown for llms)?\s*$"),
    re.compile(r"(?i)^\s*(openai|anthropic|github|discord|instagram|linkedin|select theme|dark|light|auto)\s*$"),
    re.compile(r"(?i)^\s*[\-*+]\s*\[[^\]]+\]\(#.+\)\s*$"),
]


@dataclass
class RawDocument:
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict[str, Any]


def extract_urls_from_markdown(markdown_text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]|>]+", markdown_text)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        cleaned = url.rstrip("|.,")
        if cleaned not in seen:
            deduped.append(cleaned)
            seen.add(cleaned)
    return deduped


def strip_html_tags(raw_html: str) -> str:
    without_script = re.sub(
        r"<script[\s\S]*?</script>|<style[\s\S]*?</style>",
        " ",
        raw_html,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", without_script)
    return html.unescape(text)


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove common markdown/link markup and leftover inline HTML.
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"<https?://[^>]+>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`{1,3}", "", text)

    cleaned_lines: list[str] = []
    previous_line = ""
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            previous_line = ""
            continue

        line = re.sub(r"(?i)\s*section titled\s+[\"“”][^\"“”]+[\"“”]\s*", "", line).strip()
        if not line:
            previous_line = ""
            continue

        if any(pattern.match(line) for pattern in BOILERPLATE_LINE_PATTERNS):
            previous_line = line
            continue

        # Drop table-heavy or link-farm lines that carry weak semantic signal.
        if line.count("|") >= 3 or line.count("http") >= 2:
            previous_line = line
            continue

        if any(token in line.lower() for token in ("utm_source=", "imgur.com", "<details", "</details")):
            previous_line = line
            continue

        nav_token_hits = sum(
            token in line.lower()
            for token in [
                "skip to content",
                "copy page",
                "ask questions about this page",
                "open in chatgpt",
                "open in claude",
                "select theme",
                "on this page",
                "table of contents",
            ]
        )
        if nav_token_hits >= 2:
            previous_line = line
            continue

        if line == previous_line:
            continue

        cleaned_lines.append(re.sub(r"[ \t]{2,}", " ", line))
        previous_line = line

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def flatten_json_to_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        value_dict = cast(JSONDict, value)
        for key, nested in value_dict.items():
            nested_text = flatten_json_to_text(nested)
            if nested_text:
                parts.append(f"{key}: {nested_text}")
        return "\n".join(parts)
    if isinstance(value, list):
        value_list = cast(list[Any], value)
        return "\n".join(part for item in value_list if (part := flatten_json_to_text(item)))
    if value is None:
        return ""
    return str(value)


def read_local_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md", ".rst"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix in {".html", ".htm"}:
        return strip_html_tags(file_path.read_text(encoding="utf-8", errors="ignore"))

    if suffix == ".json":
        parsed = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
        return flatten_json_to_text(parsed)

    if suffix == ".csv":
        lines: list[str] = []
        with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                lines.append(" | ".join(cell.strip() for cell in row))
        return "\n".join(lines)

    if suffix == ".pdf":
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pdfplumber is required to ingest PDF files. Install with: pip install pdfplumber"
            ) from exc

        pages: list[str] = []
        with pdfplumber.open(file_path) as pdf:  # type: ignore[attr-defined]
            pdf_any = cast(Any, pdf)
            for page in cast(list[Any], pdf_any.pages):
                extract = getattr(page, "extract_text", None)
                text = extract() if callable(extract) else ""
                pages.append(str(text or ""))
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def load_local_documents(documents_dir: Path) -> list[RawDocument]:
    docs: list[RawDocument] = []
    if not documents_dir.exists():
        return docs

    for file_path in sorted(documents_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            raw_text = read_local_file(file_path)
            if not raw_text.strip():
                continue

            relative = file_path.relative_to(documents_dir)
            docs.append(
                RawDocument(
                    doc_id=f"local::{relative.as_posix()}",
                    source="local_file",
                    title=file_path.stem,
                    text=raw_text,
                    metadata={
                        "path": str(file_path),
                        "relative_path": relative.as_posix(),
                        "extension": file_path.suffix.lower(),
                    },
                )
            )
        except Exception as exc:
            print(f"[warn] Skipping {file_path}: {exc}", file=sys.stderr)

    return docs


def fetch_url_text(url: str, timeout_seconds: int = 20) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; unofficial-guide-ingestor/1.0)"
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset, errors="ignore")

    title_match = re.search(r"<title[^>]*>(.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL)
    title = clean_text(strip_html_tags(title_match.group(1))) if title_match else url
    cleaned_text = clean_text(strip_html_tags(body))
    return title or url, cleaned_text


def fetch_text_url(url: str, timeout_seconds: int = 20) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; unofficial-guide-ingestor/1.0)"
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")


def fetch_github_repo_readme(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return None

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        return None

    owner, repo = parts[0], parts[1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"

    try:
        body = fetch_text_url(api_url)
        payload = json.loads(body)
        download_url = payload.get("download_url")
        if not download_url:
            return None

        readme_text = fetch_text_url(download_url)
        title = f"{owner}/{repo} README"
        return title, clean_text(readme_text)
    except Exception:
        return None


def fetch_reddit_thread_text(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if "reddit.com" not in parsed.netloc.lower():
        return None

    json_url = url
    if not json_url.endswith(".json"):
        json_url = json_url.rstrip("/") + ".json"

    try:
        body = fetch_text_url(json_url)
        payload = json.loads(body)
        if not isinstance(payload, list):
            return None

        payload_list = cast(list[Any], payload)
        if len(payload_list) < 2:
            return None
        post_listing = cast(JSONDict, payload_list[0]) if isinstance(payload_list[0], dict) else {}
        comments_listing = cast(JSONDict, payload_list[1]) if isinstance(payload_list[1], dict) else {}

        post_data_wrapper = cast(JSONDict, post_listing.get("data", {}))
        post_children = cast(list[Any], post_data_wrapper.get("children", []))
        if not post_children:
            return None

        first_post = cast(JSONDict, post_children[0]) if isinstance(post_children[0], dict) else {}
        post_data = cast(JSONDict, first_post.get("data", {}))
        raw_title = post_data.get("title", url)
        raw_selftext = post_data.get("selftext", "")
        title = str(raw_title)
        selftext = html.unescape(str(raw_selftext))

        comments: list[str] = []
        comments_data = cast(JSONDict, comments_listing.get("data", {}))
        for child in cast(list[Any], comments_data.get("children", [])):
            child_dict = cast(JSONDict, child) if isinstance(child, dict) else {}
            data = cast(JSONDict, child_dict.get("data", {}))
            body_text = data.get("body")
            if body_text:
                comments.append(html.unescape(str(body_text)))

        text_parts = [title, selftext]
        if comments:
            text_parts.append("Top comments:")
            text_parts.extend(comments)

        combined = "\n\n".join(part for part in text_parts if part and part.strip())
        return title, clean_text(combined)
    except Exception:
        return None


def load_url_documents(urls: Iterable[str]) -> list[RawDocument]:
    docs: list[RawDocument] = []
    for url in urls:
        try:
            parsed = urlparse(url)
            title_text: tuple[str, str] | None = None

            if parsed.netloc.lower() == "github.com":
                title_text = fetch_github_repo_readme(url)
            elif "reddit.com" in parsed.netloc.lower():
                title_text = fetch_reddit_thread_text(url)

            if title_text is None:
                title_text = fetch_url_text(url)

            title, text = title_text
            if not text.strip():
                continue

            parsed = urlparse(url)
            doc_key = f"{parsed.netloc}{parsed.path}".strip("/") or parsed.netloc
            doc_id = f"url::{doc_key}"
            docs.append(
                RawDocument(
                    doc_id=doc_id,
                    source="url",
                    title=title,
                    text=text,
                    metadata={"url": url, "domain": parsed.netloc},
                )
            )
            print(f"[info] Loaded URL: {url}")
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            print(f"[warn] Failed URL {url}: {exc}", file=sys.stderr)

    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = end - overlap

    # Re-center tiny tail chunk by rebuilding the final window while respecting max chunk size.
    if len(chunks) >= 2 and len(chunks[-1]) < max(100, chunk_size // 3):
        last_window_start = max(0, text_len - chunk_size)
        rebalanced_last = text[last_window_start:text_len].strip()
        if rebalanced_last and rebalanced_last != chunks[-2]:
            chunks[-1] = rebalanced_last

    return chunks


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_records(docs: list[RawDocument], chunk_size: int, overlap: int) -> tuple[list[JSONDict], list[JSONDict], list[JSONDict]]:
    raw_docs: list[JSONDict] = []
    clean_docs: list[JSONDict] = []
    chunk_records: list[JSONDict] = []

    for doc in docs:
        raw_docs.append(
            {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "title": doc.title,
                "text": doc.text,
                "metadata": doc.metadata,
                "char_count": len(doc.text),
            }
        )

        cleaned_text = clean_text(doc.text)
        if not cleaned_text:
            continue

        clean_docs.append(
            {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "title": doc.title,
                "text": cleaned_text,
                "metadata": doc.metadata,
                "raw_char_count": len(doc.text),
                "char_count": len(cleaned_text),
            }
        )

        chunks = chunk_text(cleaned_text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "chunk_id": f"{doc.doc_id}::chunk_{idx}",
                    "doc_id": doc.doc_id,
                    "chunk_index": idx,
                    "text": chunk,
                    "char_count": len(chunk),
                    "metadata": doc.metadata,
                }
            )

    return raw_docs, clean_docs, chunk_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents and build overlapping chunks.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--planning-file", type=Path, default=Path("planning.md"))
    parser.add_argument("--documents-dir", type=Path, default=Path("documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    parser.add_argument("--include-urls", action="store_true", default=True)
    parser.add_argument("--no-include-urls", dest="include_urls", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    planning_file = (project_root / args.planning_file).resolve()
    documents_dir = (project_root / args.documents_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if args.chunk_size < 800 or args.chunk_size > 1000:
        print("[warn] Chunk size is outside planning target range (800-1000).", file=sys.stderr)
    if args.chunk_overlap < 150 or args.chunk_overlap > 200:
        print("[warn] Chunk overlap is outside planning target range (150-200).", file=sys.stderr)

    if args.chunk_overlap >= args.chunk_size:
        print("[error] chunk-overlap must be smaller than chunk-size.", file=sys.stderr)
        return 1

    local_docs = load_local_documents(documents_dir)
    print(f"[info] Loaded local docs: {len(local_docs)}")

    url_docs: list[RawDocument] = []
    if args.include_urls and planning_file.exists():
        planning_text = planning_file.read_text(encoding="utf-8", errors="ignore")
        urls = extract_urls_from_markdown(planning_text)
        print(f"[info] URLs discovered in planning file: {len(urls)}")
        url_docs = load_url_documents(urls)

    all_docs = local_docs + url_docs
    if not all_docs:
        print("[warn] No documents were loaded. Add files under documents/ or enable URL ingestion.")
        return 0



    raw_docs, clean_docs, chunk_records = build_records(all_docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_docs_path = output_dir / "raw_documents.jsonl"
    clean_docs_path = output_dir / "clean_documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    write_jsonl(raw_docs_path, raw_docs)
    write_jsonl(clean_docs_path, clean_docs)
    write_jsonl(chunks_path, chunk_records)

    print(f"[info] Wrote {len(raw_docs)} raw documents -> {raw_docs_path}")
    print(f"[info] Wrote {len(clean_docs)} cleaned documents -> {clean_docs_path}")
    print(f"[info] Wrote {len(chunk_records)} chunks -> {chunks_path}")

    if chunk_records:
        avg_len = sum(item["char_count"] for item in chunk_records) / len(chunk_records)
        print(f"[info] Average chunk length: {avg_len:.1f} chars")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
