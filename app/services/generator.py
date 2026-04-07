"""LLM generation service for RAG answers."""

import logging

from openai import OpenAI

from app.core.config import settings
from app.models.schemas import RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一个知识库问答助手。根据检索到的图文内容回答用户的问题。\n"
    "回答要求：\n"
    "1. 仅根据提供的检索结果回答，不要编造信息\n"
    "2. 如果检索结果不足以回答问题，明确告知用户\n"
    "3. 引用具体的图片文件名作为依据\n"
    "4. 使用中文回答"
)

QUERY_REWRITE_PROMPT = (
    "你是一个查询理解助手。用户会用中文描述他们想找的图片，你需要将其改写为简洁的英文描述，用于图像检索。\n"
    "要求：\n"
    "1. 提取核心视觉元素（主体、动作、场景、颜色等）\n"
    "2. 使用简单的英文短语，不要完整句子\n"
    "3. 只输出英文描述，不要解释\n"
    "4. 如果用户查询已经是英文，直接返回原文\n\n"
    "示例：\n"
    "用户：一只在草地上奔跑的狗\n"
    "输出：dog running on grass\n\n"
    "用户：穿红色衣服的小女孩\n"
    "输出：little girl in red dress\n\n"
    "用户：海边的日落\n"
    "输出：sunset at beach"
)


def _format_sources(sources: list[RetrievalResult]) -> str:
    """Format retrieval results into context string for the LLM."""
    parts = []
    for i, src in enumerate(sources, 1):
        captions_text = "; ".join(src.captions) if src.captions else "无描述"
        parts.append(
            f"[来源 {i}] 文件: {src.filename} (相似度: {src.score:.3f})\n"
            f"描述: {captions_text}"
        )
    return "\n\n".join(parts)


class LLMGenerator:
    """Generates answers using an OpenAI-compatible LLM API."""

    def __init__(self) -> None:
        if not settings.llm_api_key:
            raise ValueError(
                "LLM_API_KEY is not set. "
                "Please configure it in .env file."
            )
        self._client = OpenAI(
            base_url=settings.llm_base_url or None,
            api_key=settings.llm_api_key,
        )
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    def generate(self, query: str, sources: list[RetrievalResult]) -> str:
        """Generate an answer based on query and retrieved sources."""
        context = _format_sources(sources)
        user_message = (
            f"检索到的相关内容：\n{context}\n\n"
            f"用户问题：{query}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content

    def rewrite_query(self, query: str) -> str:
        """Rewrite user query to English for better CLIP retrieval."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=100,
            temperature=0.3,
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info("Query rewrite: '%s' -> '%s'", query, rewritten)
        return rewritten
