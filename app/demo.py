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
        return [], "请输入查询文本"
    try:
        resp = retriever.text_to_image(query.strip(), int(top_k))
        gallery = []
        info_lines = []
        for i, r in enumerate(resp.results, 1):
            path = Path(r.filepath)
            if path.exists():
                gallery.append(str(path))
            info_lines.append(
                f"**#{i}** {r.filename}  \n"
                f"相似度: {r.score:.4f}  \n"
                f"Captions: {' | '.join(r.captions[:2])}"
            )
        summary = (
            f"**查询**: {query}  \n"
            f"**返回**: {len(resp.results)} 张图片  \n"
            f"**索引总量**: {resp.total_indexed}  \n"
            f"**耗时**: {resp.elapsed_ms:.1f} ms\n\n---\n\n"
            + "\n\n".join(info_lines)
        )
        return gallery, summary
    except Exception as e:
        logger.exception("text_to_image failed")
        return [], f"**错误**: {e}"


# ── Tab 2: Image → Image ─────────────────────────────────────────────

def image_to_image(image_path: str, top_k: int):
    if not image_path:
        return [], "请上传一张图片"
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
                f"相似度: {r.score:.4f}  \n"
                f"Captions: {' | '.join(r.captions[:2])}"
            )
        summary = (
            f"**返回**: {len(resp.results)} 张相似图片  \n"
            f"**耗时**: {resp.elapsed_ms:.1f} ms\n\n---\n\n"
            + "\n\n".join(info_lines)
        )
        return gallery, summary
    except Exception as e:
        logger.exception("image_to_image failed")
        return [], f"**错误**: {e}"


# ── Tab 3: RAG Q&A ───────────────────────────────────────────────────

def rag_query(query: str, top_k: int):
    if not query.strip():
        return "请输入问题", [], ""
    try:
        resp = retriever.rag_query(query.strip(), int(top_k), generator)
        gallery = []
        source_lines = []
        for i, s in enumerate(resp.sources, 1):
            path = Path(s.filepath)
            if path.exists():
                gallery.append(str(path))
            source_lines.append(f"**#{i}** {s.filename} (相似度: {s.score:.4f})")
        timing = (
            f"检索耗时: {resp.retrieval_ms:.1f} ms | "
            f"生成耗时: {resp.generation_ms:.1f} ms"
        )
        sources_text = "\n\n".join(source_lines) if source_lines else "无来源"
        return resp.answer, gallery, f"{timing}\n\n---\n\n{sources_text}"
    except Exception as e:
        logger.exception("rag_query failed")
        return f"错误: {e}", [], ""


# ── Tab 4: System Status ─────────────────────────────────────────────

def get_status():
    n_images = metadata.count_images()
    n_captions = metadata.count_captions()
    img_idx = indexer.image_index_size
    txt_idx = indexer.text_index_size
    return (
        f"## 系统状态\n\n"
        f"| 指标 | 数值 |\n"
        f"|------|------|\n"
        f"| 图片索引向量数 | {img_idx} |\n"
        f"| 文本索引向量数 | {txt_idx} |\n"
        f"| 元数据图片数 | {n_images} |\n"
        f"| 元数据 Caption 数 | {n_captions} |\n"
        f"| 嵌入维度 | 512 |\n"
        f"| CLIP 模型 | ViT-B/32 |\n"
        f"| 索引类型 | IndexFlatIP (精确) |\n"
        f"| 设备 | {settings.device} |\n"
        f"| 图片目录 | `{settings.flickr30k_image_dir}` |\n"
    )


# ── Build UI ──────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Multimodal RAG Demo") as app:
        gr.Markdown("# 🔍 多模态检索增强生成系统\nCLIP + FAISS + LLM")

        with gr.Tab("文本搜图"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_query = gr.Textbox(
                        label="查询文本",
                        placeholder="e.g. a dog playing on grass",
                        lines=2,
                    )
                    t2i_topk = gr.Slider(1, 20, value=5, step=1, label="Top K")
                    t2i_btn = gr.Button("搜索", variant="primary")
                with gr.Column(scale=2):
                    t2i_gallery = gr.Gallery(
                        label="检索结果", columns=3, height=400
                    )
                    t2i_info = gr.Markdown()
            t2i_btn.click(text_to_image, [t2i_query, t2i_topk], [t2i_gallery, t2i_info])
            t2i_query.submit(text_to_image, [t2i_query, t2i_topk], [t2i_gallery, t2i_info])

        with gr.Tab("以图搜图"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_input = gr.Image(label="上传查询图片", type="filepath")
                    i2i_topk = gr.Slider(1, 20, value=5, step=1, label="Top K")
                    i2i_btn = gr.Button("搜索", variant="primary")
                with gr.Column(scale=2):
                    i2i_gallery = gr.Gallery(
                        label="相似图片", columns=3, height=400
                    )
                    i2i_info = gr.Markdown()
            i2i_btn.click(image_to_image, [i2i_input, i2i_topk], [i2i_gallery, i2i_info])

        with gr.Tab("RAG 问答"):
            with gr.Row():
                with gr.Column(scale=1):
                    rag_input = gr.Textbox(
                        label="问题",
                        placeholder="e.g. describe images with children playing",
                        lines=2,
                    )
                    rag_topk = gr.Slider(1, 10, value=5, step=1, label="检索数量")
                    rag_btn = gr.Button("提问", variant="primary")
                with gr.Column(scale=2):
                    rag_answer = gr.Textbox(label="LLM 回答", lines=8, interactive=False)
                    rag_gallery = gr.Gallery(label="检索来源", columns=3, height=300)
                    rag_meta = gr.Markdown()
            rag_btn.click(rag_query, [rag_input, rag_topk], [rag_answer, rag_gallery, rag_meta])
            rag_input.submit(rag_query, [rag_input, rag_topk], [rag_answer, rag_gallery, rag_meta])

        with gr.Tab("系统状态"):
            status_md = gr.Markdown()
            refresh_btn = gr.Button("刷新状态")
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
