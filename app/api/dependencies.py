"""Dependency injection: singleton services for FastAPI."""

from functools import lru_cache

from app.services.encoder import CLIPEncoder
from app.services.generator import LLMGenerator
from app.services.indexer import FAISSIndexer
from app.services.metadata import MetadataStore
from app.services.retriever import Retriever


@lru_cache(maxsize=1)
def get_encoder() -> CLIPEncoder:
    return CLIPEncoder()


@lru_cache(maxsize=1)
def get_indexer() -> FAISSIndexer:
    indexer = FAISSIndexer()
    indexer.load()
    return indexer


@lru_cache(maxsize=1)
def get_metadata() -> MetadataStore:
    return MetadataStore()


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(
        encoder=get_encoder(),
        indexer=get_indexer(),
        metadata=get_metadata(),
    )


@lru_cache(maxsize=1)
def get_generator() -> LLMGenerator:
    return LLMGenerator()
