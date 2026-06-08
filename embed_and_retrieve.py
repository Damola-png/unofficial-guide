#!/usr/bin/env python3
"""Embed chunked documents into ChromaDB and retrieve relevant chunks.

This script implements Milestone 4 from planning.md:
- Embedding model: all-MiniLM-L6-v2
- Retrieval: top-k nearest chunks from ChromaDB
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_EVAL_QUERIES = [
    "When should students start applying for CS internships?",
    "What should a CS internship resume include?",
    "Do referrals help compared to cold applications?",
    "What should students prepare for online assessments?",
    "What advice do students give for behavioral interviews?",
]


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    # Runs fully local with sentence-transformers; no API key or rate limits required.
    return SentenceTransformer(model_name)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _safe_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=True)


def build_chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    source_meta = chunk.get("metadata", {})
    source_name = str(chunk.get("doc_id", "")).strip()
    if isinstance(source_meta, dict):
        for candidate in ("relative_path", "path", "url", "domain"):
            value = source_meta.get(candidate)
            if value:
                source_name = str(value)
                break

    metadata: dict[str, str | int | float | bool] = {
        "doc_id": _safe_value(chunk.get("doc_id", "")),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "chunk_position": int(chunk.get("chunk_index", 0)),
        "char_count": int(chunk.get("char_count", 0)),
        "chunk_id": _safe_value(chunk.get("chunk_id", "")),
        "source_document_name": _safe_value(source_name),
    }

    if isinstance(source_meta, dict):
        for key, value in source_meta.items():
            metadata[f"source_{key}"] = _safe_value(value)

    return metadata


def extract_source_info(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize source fields so retrieval callers can cite where chunks came from."""
    if not isinstance(metadata, dict):
        return {"source": "unknown"}

    source = (
        metadata.get("source_document_name")
        or metadata.get("source_relative_path")
        or metadata.get("source_url")
        or metadata.get("source_path")
        or metadata.get("doc_id")
        or "unknown"
    )

    source_info: dict[str, Any] = {"source": source}
    for key in ("source_url", "source_domain", "source_relative_path", "source_path", "doc_id"):
        if key in metadata and metadata[key] not in (None, ""):
            source_info[key] = metadata[key]
    return source_info


def embed_chunks(
    chunks_path: Path,
    chroma_dir: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
    reset_collection: bool,
    distance_space: str,
) -> int:
    chunks = read_jsonl(chunks_path)
    if not chunks:
        print("[warn] No chunks found. Nothing to embed.")
        return 0

    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not text or not chunk_id:
            continue
        texts.append(text)
        ids.append(chunk_id)
        metadatas.append(build_chroma_metadata(chunk))

    if not texts:
        print("[warn] Chunks are present, but none had valid text/chunk_id.")
        return 0

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if reset_collection:
        try:
            client.delete_collection(name=collection_name)
            print(f"[info] Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_space},
    )

    model = load_embedding_model(model_name)

    total = len(texts)
    print(f"[info] Embedding {total} chunks with {model_name}...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_metadatas = metadatas[start:end]

        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # upsert makes this idempotent: existing chunk_id rows are updated, new ones inserted.
        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas,
            embeddings=embeddings.tolist(),
        )
        print(f"[info] Upserted {end}/{total}")

    print(f"[info] Collection '{collection_name}' now has {collection.count()} vectors")
    return total


def retrieve(
    query: str,
    chroma_dir: Path,
    collection_name: str,
    model_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=collection_name)

    model = load_embedding_model(model_name)
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns list-of-lists because query() supports multiple query vectors at once.
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    rows: list[dict[str, Any]] = []
    for idx, chunk_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        rows.append(
            {
                "rank": idx + 1,
                "chunk_id": chunk_id,
                "text": docs[idx] if idx < len(docs) else "",
                "metadata": metadata,
                "source": extract_source_info(metadata),
                "distance": distances[idx] if idx < len(distances) else None,
            }
        )
    return rows


def evaluate_retrieval(
    queries: list[str],
    chroma_dir: Path,
    collection_name: str,
    model_name: str,
    top_k: int,
) -> None:
    for query in queries:
        print(f"\n=== Query: {query} ===")
        rows = retrieve(
            query=query,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            model_name=model_name,
            top_k=top_k,
        )
        if not rows:
            print("[warn] No results returned.")
            continue

        best_distance = rows[0].get("distance")
        if isinstance(best_distance, (int, float)) and best_distance > 0.7:
            print("[warn] Best distance is high (> 0.7): likely weak retrieval signal.")

        for row in rows:
            distance = row.get("distance")
            source = row.get("source", {}).get("source", "unknown")
            print(f"\n[rank {row['rank']}] distance={distance} source={source}")
            print(f"chunk_id={row.get('chunk_id', '')}")
            print("--- chunk text start ---")
            print(str(row.get("text", "")))
            print("--- chunk text end ---")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunks and retrieve relevant context.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--chroma-dir", type=Path, default=Path("artifacts/chroma"))
    parser.add_argument("--collection-name", type=str, default="internship_guide_chunks")

    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Embed chunks.jsonl into ChromaDB")
    embed_parser.add_argument("--chunks-path", type=Path, default=Path("artifacts/chunks.jsonl"))
    embed_parser.add_argument("--batch-size", type=int, default=64)
    embed_parser.add_argument("--reset-collection", action="store_true", help="Delete and recreate collection before embedding")
    embed_parser.add_argument(
        "--distance-space",
        type=str,
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="Chroma distance metric used by the collection",
    )

    query_parser = subparsers.add_parser("query", help="Run retrieval against ChromaDB")
    query_parser.add_argument("--q", type=str, required=True, help="Search query")
    query_parser.add_argument("--top-k", type=int, default=5)

    eval_parser = subparsers.add_parser("eval", help="Run retrieval on planned evaluation queries")
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--count", type=int, default=3, help="Number of default queries to run")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    chroma_dir = (project_root / args.chroma_dir).resolve()

    if args.command == "embed":
        chunks_path = (project_root / args.chunks_path).resolve()
        embed_chunks(
            chunks_path=chunks_path,
            chroma_dir=chroma_dir,
            collection_name=args.collection_name,
            model_name=args.model_name,
            batch_size=args.batch_size,
            reset_collection=args.reset_collection,
            distance_space=args.distance_space,
        )
        return 0

    if args.command == "query":
        rows = retrieve(
            query=args.q,
            chroma_dir=chroma_dir,
            collection_name=args.collection_name,
            model_name=args.model_name,
            top_k=args.top_k,
        )
        print(json.dumps(rows, indent=2, ensure_ascii=True))
        return 0

    if args.command == "eval":
        count = max(1, min(args.count, len(DEFAULT_EVAL_QUERIES)))
        evaluate_retrieval(
            queries=DEFAULT_EVAL_QUERIES[:count],
            chroma_dir=chroma_dir,
            collection_name=args.collection_name,
            model_name=args.model_name,
            top_k=args.top_k,
        )
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
