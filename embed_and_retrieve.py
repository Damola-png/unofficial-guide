#!/usr/bin/env python3
"""Embed chunked documents into ChromaDB and retrieve relevant chunks.

This script implements Milestone 4 from planning.md:
- Embedding model: all-MiniLM-L6-v2
- Retrieval: top-k nearest chunks from ChromaDB
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


DEFAULT_EVAL_QUERIES = [
    "When should students start applying for CS internships?",
    "What should a CS internship resume include?",
    "Do referrals help compared to cold applications?",
    "What should students prepare for online assessments?",
    "What advice do students give for behavioral interviews?",
]

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_MAX_CONTEXT_CHARS = 7000


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


def format_retrieval_context(rows: list[dict[str, Any]], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> tuple[str, list[dict[str, Any]]]:
    """Build a bounded context block for the LLM and return the used rows."""
    context_parts: list[str] = []
    used_rows: list[dict[str, Any]] = []
    total_chars = 0

    for row in rows:
        source = row.get("source", {}).get("source", "unknown")
        distance = row.get("distance")
        header = f"[S{row['rank']}] source={source} distance={distance}\n"
        body = str(row.get("text", "")).strip()
        block = f"{header}{body}\n"
        if not body:
            continue
        if context_parts and total_chars + len(block) > max_chars:
            break
        context_parts.append(block)
        used_rows.append(row)
        total_chars += len(block)

    return "\n".join(context_parts).strip(), used_rows


def build_grounding_messages(query: str, context_text: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are a grounded RAG assistant for CS internship advice. "
        "Answer ONLY using the provided retrieved context. "
        "If the context is insufficient or missing, say you do not have enough evidence in the retrieved documents. "
        "Do not use outside knowledge. "
        "When making claims, cite source labels like [S1], [S2]."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Write a concise answer grounded in the context and include citations."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_grounded_answer(
    query: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    top_k: int,
    distance_threshold: float,
    llm_model: str,
    max_context_chars: int,
) -> dict[str, Any]:
    rows = retrieve(
        query=query,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        model_name=embedding_model_name,
        top_k=top_k,
    )

    filtered_rows = [
        row
        for row in rows
        if isinstance(row.get("distance"), (int, float)) and float(row["distance"]) <= distance_threshold
    ]
    rows_for_generation = filtered_rows if filtered_rows else rows[: min(2, len(rows))]

    context_text, used_rows = format_retrieval_context(rows_for_generation, max_chars=max_context_chars)
    if not context_text:
        return {
            "answer": "I do not have enough evidence in the retrieved documents to answer this question.",
            "retrieval": rows,
            "used_chunks": [],
            "low_confidence": True,
        }

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to .env before running generation.")

    client = Groq(api_key=api_key)
    messages = build_grounding_messages(query=query, context_text=context_text)
    completion = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=0.2,
    )
    answer = completion.choices[0].message.content or ""

    return {
        "answer": answer.strip(),
        "retrieval": rows,
        "used_chunks": used_rows,
        "low_confidence": len(filtered_rows) == 0,
    }


def print_generation_result(result: dict[str, Any]) -> None:
    print("\n=== Grounded Answer ===")
    print(result.get("answer", ""))

    if result.get("low_confidence"):
        print("\n[warn] No chunks met distance threshold. Answer used fallback top chunks.")

    used_chunks = result.get("used_chunks", [])
    if isinstance(used_chunks, list) and used_chunks:
        print("\n=== Sources Used ===")
        for row in used_chunks:
            source = row.get("source", {}).get("source", "unknown")
            print(f"[S{row.get('rank')}] source={source} distance={row.get('distance')}")


def run_chat_interface(
    chroma_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    top_k: int,
    distance_threshold: float,
    llm_model: str,
    max_context_chars: int,
) -> int:
    print("Grounded RAG chat is ready. Type 'exit' or 'quit' to stop.")
    while True:
        query = input("\nAsk a question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = generate_grounded_answer(
            query=query,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            top_k=top_k,
            distance_threshold=distance_threshold,
            llm_model=llm_model,
            max_context_chars=max_context_chars,
        )
        print_generation_result(result)

    return 0


def run_gradio_interface(
    chroma_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    top_k: int,
    distance_threshold: float,
    llm_model: str,
    max_context_chars: int,
    host: str,
    port: int,
) -> int:
    try:
        import gradio as gr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Install with: pip install gradio"
        ) from exc

    def _answer(question: str) -> tuple[str, str]:
        question = question.strip()
        if not question:
            return "Please enter a question.", ""

        result = generate_grounded_answer(
            query=question,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            top_k=top_k,
            distance_threshold=distance_threshold,
            llm_model=llm_model,
            max_context_chars=max_context_chars,
        )

        source_lines: list[str] = []
        used_chunks = result.get("used_chunks", [])
        if isinstance(used_chunks, list):
            for row in used_chunks:
                source = row.get("source", {}).get("source", "unknown")
                source_lines.append(f"[S{row.get('rank')}] source={source} distance={row.get('distance')}")

        return str(result.get("answer", "")), "\n".join(source_lines)

    ui = gr.Interface(
        fn=_answer,
        inputs=gr.Textbox(lines=2, label="Question"),
        outputs=[
            gr.Textbox(lines=10, label="Grounded answer"),
            gr.Textbox(lines=6, label="Sources used"),
        ],
        title="Unofficial Guide RAG",
        description="Answers are generated from retrieved chunks only and include source labels.",
    )
    ui.launch(server_name=host, server_port=port)
    return 0


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

    ask_parser = subparsers.add_parser("ask", help="Generate grounded answer for one question")
    ask_parser.add_argument("--q", type=str, required=True, help="Question to answer")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--distance-threshold", type=float, default=0.5)
    ask_parser.add_argument("--llm-model", type=str, default=DEFAULT_GROQ_MODEL)
    ask_parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)

    chat_parser = subparsers.add_parser("chat", help="Open terminal chat interface")
    chat_parser.add_argument("--top-k", type=int, default=5)
    chat_parser.add_argument("--distance-threshold", type=float, default=0.5)
    chat_parser.add_argument("--llm-model", type=str, default=DEFAULT_GROQ_MODEL)
    chat_parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)

    serve_parser = subparsers.add_parser("serve", help="Run a Gradio web interface")
    serve_parser.add_argument("--top-k", type=int, default=5)
    serve_parser.add_argument("--distance-threshold", type=float, default=0.5)
    serve_parser.add_argument("--llm-model", type=str, default=DEFAULT_GROQ_MODEL)
    serve_parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=7860)

    return parser.parse_args()


def main() -> int:
    load_dotenv()
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

    if args.command == "ask":
        result = generate_grounded_answer(
            query=args.q,
            chroma_dir=chroma_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.model_name,
            top_k=args.top_k,
            distance_threshold=args.distance_threshold,
            llm_model=args.llm_model,
            max_context_chars=args.max_context_chars,
        )
        print_generation_result(result)
        return 0

    if args.command == "chat":
        return run_chat_interface(
            chroma_dir=chroma_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.model_name,
            top_k=args.top_k,
            distance_threshold=args.distance_threshold,
            llm_model=args.llm_model,
            max_context_chars=args.max_context_chars,
        )

    if args.command == "serve":
        return run_gradio_interface(
            chroma_dir=chroma_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.model_name,
            top_k=args.top_k,
            distance_threshold=args.distance_threshold,
            llm_model=args.llm_model,
            max_context_chars=args.max_context_chars,
            host=args.host,
            port=args.port,
        )

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
