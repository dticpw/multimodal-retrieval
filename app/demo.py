"""Gradio Demo: multimodal retrieval system interactive interface.

Usage:
    python -m app.demo
"""

import logging
import os
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from app.api.dependencies import get_generator, get_indexer, get_metadata, get_retriever
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

retriever = get_retriever()
indexer = get_indexer()
metadata = get_metadata()
generator = get_generator()


# ── Tab 1: Text → Image ──────────────────────────────────────────────

def text_to_image(query: str, top_k: int):
    if not query.strip():
        return [], "Please enter a query text"
    try:
        resp = retriever.text_to_image(query.strip(), int(top_k), generator=generator)
        gallery = []
        info_lines = []
        for i, r in enumerate(resp.results, 1):
            path = Path(r.filepath)
            if path.exists():
                gallery.append(str(path))
            info_lines.append(
                f"**#{i}** {r.filename}  \n"
                f"Similarity: {r.score:.4f}  \n"
                f"Captions: {' | '.join(r.captions[:2])}"
            )
        summary = (
            f"**Query**: {query}  \n"
            f"**Results**: {len(resp.results)} images  \n"
            f"**Total Indexed**: {resp.total_indexed}  \n"
            f"**Latency**: {resp.elapsed_ms:.1f} ms\n\n---\n\n"
            + "\n\n".join(info_lines)
        )
        return gallery, summary
    except Exception as e:
        logger.exception("text_to_image failed")
        return [], f"**Error**: {e}"


# ── Tab 2: Image → Image ─────────────────────────────────────────────

def image_to_image(image_path: str, top_k: int):
    if not image_path:
        return [], "Please upload an image"
    try:
        resp = retriever.image_to_image(image_path, int(top_k))
        gallery = []
        info_lines = []
        for i, r in enumerate(resp.results, 1):
            path = Path(r.filepath)
            if path.exists():
                gallery.append(str(path))
            info_lines.append(
                f"**#{i}** {r.filename}  \n"
                f"Similarity: {r.score:.4f}  \n"
                f"Captions: {' | '.join(r.captions[:2])}"
            )
        summary = (
            f"**Results**: {len(resp.results)} similar images  \n"
            f"**Latency**: {resp.elapsed_ms:.1f} ms\n\n---\n\n"
            + "\n\n".join(info_lines)
        )
        return gallery, summary
    except Exception as e:
        logger.exception("image_to_image failed")
        return [], f"**Error**: {e}"


# ── Tab 3: RAG Q&A ───────────────────────────────────────────────────

def rag_query(query: str, top_k: int):
    if not query.strip():
        return "Please enter a question", [], ""
    try:
        resp = retriever.rag_query(query.strip(), int(top_k), generator)
        gallery = []
        source_lines = []
        for i, s in enumerate(resp.sources, 1):
            path = Path(s.filepath)
            if path.exists():
                gallery.append(str(path))
            source_lines.append(f"**#{i}** {s.filename} (Similarity: {s.score:.4f})")
        timing = (
            f"Retrieval: {resp.retrieval_ms:.1f} ms | "
            f"Generation: {resp.generation_ms:.1f} ms"
        )
        sources_text = "\n\n".join(source_lines) if source_lines else "No sources"
        return resp.answer, gallery, f"{timing}\n\n---\n\n{sources_text}"
    except Exception as e:
        logger.exception("rag_query failed")
        return f"Error: {e}", [], ""


# ── Tab 4: System Status ─────────────────────────────────────────────

def get_status():
    n_images = metadata.count_images()
    n_captions = metadata.count_captions()
    img_idx = indexer.image_index_size
    txt_idx = indexer.text_index_size
    return (
        f"## System Status\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Image Index Vectors | {img_idx} |\n"
        f"| Text Index Vectors | {txt_idx} |\n"
        f"| Metadata Images | {n_images} |\n"
        f"| Metadata Captions | {n_captions} |\n"
        f"| Embedding Dimension | 512 |\n"
        f"| CLIP Model | ViT-B/32 |\n"
        f"| Index Type | IndexFlatIP (Exact) |\n"
        f"| Device | {settings.device} |\n"
        f"| Image Directory | `{settings.flickr30k_image_dir}` |\n"
    )


# ── Build UI ──────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Multimodal RAG Demo") as app:
        gr.Markdown("# 🔍 Multimodal Retrieval-Augmented Generation System\nCLIP + FAISS + LLM")

        with gr.Tab("Text-to-Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_query = gr.Textbox(
                        label="Query Text",
                        placeholder="e.g. a dog playing on grass",
                        lines=2,
                    )
                    t2i_topk = gr.Slider(1, 20, value=5, step=1, label="Top K")
                    t2i_btn = gr.Button("Search", variant="primary")
                with gr.Column(scale=2):
                    t2i_gallery = gr.Gallery(
                        label="Retrieval Results", columns=5, rows=2
                    )
                    t2i_info = gr.Markdown()
            t2i_btn.click(text_to_image, [t2i_query, t2i_topk], [t2i_gallery, t2i_info])
            t2i_query.submit(text_to_image, [t2i_query, t2i_topk], [t2i_gallery, t2i_info])

        with gr.Tab("Image-to-Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_input = gr.Image(label="Upload Query Image", type="filepath")
                    i2i_topk = gr.Slider(1, 20, value=5, step=1, label="Top K")
                    i2i_btn = gr.Button("Search", variant="primary")
                with gr.Column(scale=2):
                    i2i_gallery = gr.Gallery(
                        label="Similar Images", columns=5, rows=2
                    )
                    i2i_info = gr.Markdown()
            i2i_btn.click(image_to_image, [i2i_input, i2i_topk], [i2i_gallery, i2i_info])

        with gr.Tab("RAG Q&A"):
            with gr.Row():
                with gr.Column(scale=1):
                    rag_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g. describe images with children playing",
                        lines=2,
                    )
                    rag_topk = gr.Slider(1, 10, value=5, step=1, label="Retrieval Count")
                    rag_btn = gr.Button("Ask", variant="primary")
                with gr.Column(scale=2):
                    rag_answer = gr.Textbox(label="LLM Answer", lines=8, interactive=False)
                    rag_gallery = gr.Gallery(label="Retrieved Sources", columns=5, rows=2)
                    rag_meta = gr.Markdown()
            rag_btn.click(rag_query, [rag_input, rag_topk], [rag_answer, rag_gallery, rag_meta])
            rag_input.submit(rag_query, [rag_input, rag_topk], [rag_answer, rag_gallery, rag_meta])

        with gr.Tab("System Status"):
            status_md = gr.Markdown()
            refresh_btn = gr.Button("Refresh Status")
            refresh_btn.click(get_status, [], [status_md])
            app.load(get_status, [], [status_md])

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        allowed_paths=[settings.flickr30k_image_dir],
    )
