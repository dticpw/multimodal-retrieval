from fastapi import APIRouter, Depends

from app.api.dependencies import get_generator, get_indexer, get_metadata, get_retriever
from app.models.schemas import ImageQuery, RAGQuery, RAGResponse, RetrievalResponse, TextQuery
from app.services.generator import LLMGenerator
from app.services.indexer import FAISSIndexer
from app.services.metadata import MetadataStore
from app.services.retriever import Retriever

router = APIRouter(prefix="/api/v1")


@router.post("/search/text-to-image", response_model=RetrievalResponse)
def text_to_image(
    query: TextQuery,
    retriever: Retriever = Depends(get_retriever),
):
    return retriever.text_to_image(query.query, query.top_k)


@router.post("/search/image-to-image", response_model=RetrievalResponse)
def image_to_image(
    query: ImageQuery,
    retriever: Retriever = Depends(get_retriever),
):
    return retriever.image_to_image(query.image_path, query.top_k)


@router.post("/search/image-to-text", response_model=RetrievalResponse)
def image_to_text(
    query: ImageQuery,
    retriever: Retriever = Depends(get_retriever),
):
    return retriever.image_to_text(query.image_path, query.top_k)


@router.post("/rag/query", response_model=RAGResponse)
def rag_query(
    query: RAGQuery,
    retriever: Retriever = Depends(get_retriever),
    generator: LLMGenerator = Depends(get_generator),
):
    return retriever.rag_query(query.query, query.top_k, generator)


@router.get("/status")
def status(
    indexer: FAISSIndexer = Depends(get_indexer),
    metadata: MetadataStore = Depends(get_metadata),
):
    return {
        "status": "ok",
        "image_index_size": indexer.image_index_size,
        "text_index_size": indexer.text_index_size,
        "total_images": metadata.count_images(),
        "total_captions": metadata.count_captions(),
    }
