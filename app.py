#!/usr/bin/env python3
"""Gradio web UI for grounded retrieval-augmented generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from embed_and_retrieve import DEFAULT_GROQ_MODEL, DEFAULT_MAX_CONTEXT_CHARS, generate_grounded_answer


def build_sources_text(used_chunks: list[dict[str, Any]]) -> str:
    if not used_chunks:
        return "No sources used."

    lines: list[str] = []
    for row in used_chunks:
        source = row.get("source", {}).get("source", "unknown")
        lines.append(
            f"[S{row.get('rank')}] {source} | chunk_id={row.get('chunk_id', '')} | distance={row.get('distance')}"
        )
    return "\n".join(lines)


def make_handle_query(project_root: Path, model_name: str, collection_name: str, top_k: int, distance_threshold: float, llm_model: str, max_context_chars: int):
    def handle_query(question: str) -> tuple[str, str]:
        question = question.strip()
        if not question:
            return "Please enter a question.", ""

        result = generate_grounded_answer(
            query=question,
            chroma_dir=project_root / "artifacts" / "chroma",
            collection_name=collection_name,
            embedding_model_name=model_name,
            top_k=top_k,
            distance_threshold=distance_threshold,
            llm_model=llm_model,
            max_context_chars=max_context_chars,
        )
        return str(result.get("response_text", result.get("answer", ""))), build_sources_text(result.get("used_chunks", []))

    return handle_query


def build_ui(project_root: Path, model_name: str, collection_name: str, top_k: int, distance_threshold: float, llm_model: str, max_context_chars: int) -> gr.Blocks:
    handle_query = make_handle_query(
        project_root=project_root,
        model_name=model_name,
        collection_name=collection_name,
        top_k=top_k,
        distance_threshold=distance_threshold,
        llm_model=llm_model,
        max_context_chars=max_context_chars,
    )

    with gr.Blocks(title="Unofficial Guide RAG") as demo:
        gr.Markdown("# Unofficial Guide RAG\nAsk a question and get an answer grounded only in the retrieved documents.")
        question = gr.Textbox(label="Your question", lines=2, placeholder="e.g. When should students start applying for CS internships?")
        ask_button = gr.Button("Ask")
        answer = gr.Textbox(label="Answer", lines=10)
        sources = gr.Textbox(label="Retrieved from", lines=6)

        ask_button.click(handle_query, inputs=question, outputs=[answer, sources])
        question.submit(handle_query, inputs=question, outputs=[answer, sources])

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Gradio interface for grounded RAG.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--collection-name", type=str, default="internship_guide_chunks")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--distance-threshold", type=float, default=0.5)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    demo = build_ui(
        project_root=args.project_root.resolve(),
        model_name=args.model_name,
        collection_name=args.collection_name,
        top_k=args.top_k,
        distance_threshold=args.distance_threshold,
        llm_model=args.llm_model,
        max_context_chars=args.max_context_chars,
    )
    demo.launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
