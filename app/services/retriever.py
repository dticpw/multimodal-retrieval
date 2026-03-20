import logging
import time
from pathlib import Path

from app.models.schemas import (
    CaptionResult,
    RAGResponse,
    RetrievalResponse,
    RetrievalResult,
)
from app.services.generator import LLMGenerator
from app.services.encoder import CLIPEncoder
from app.services.indexer import FAISSIndexer
from app.services.metadata import MetadataStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates CLIP encoding, FAISS search, and metadata lookup."""

    def __init__(
        self,
        encoder: CLIPEncoder,
        indexer: FAISSIndexer,
        metadata: MetadataStore,
    ):
        self._encoder = encoder
        self._indexer = indexer
        self._metadata = metadata

    def text_to_image(self, query: str, top_k: int = 10) -> RetrievalResponse:
        """Search images by text query."""
        start = time.perf_counter()

        query_emb = self._encoder.encode_texts([query])
        scores, indices = self._indexer.search_images(query_emb, top_k)

        idx_list = indices[0].tolist()
        score_list = scores[0].tolist()

        valid_indices = [i for i in idx_list if i >= 0]
        records = self._metadata.get_images_by_indices(valid_indices)
        idx_to_record = {}
        for idx, rec in zip(valid_indices, records):
            idx_to_record[idx] = rec

        results = []
        for faiss_idx, score in zip(idx_list, score_list):
            if faiss_idx < 0 or faiss_idx not in idx_to_record:
                continue
            rec = idx_to_record[faiss_idx]
            results.append(
                RetrievalResult(
                    image_id=rec.image_id,
                    filename=rec.filename,
                    filepath=rec.filepath,
                    score=float(score),
                    captions=rec.captions,
                )
            )

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResponse(
            query_type="text_to_image",
            results=results,
            total_indexed=self._indexer.image_index_size,
            elapsed_ms=round(elapsed, 2),
        )

    def image_to_image(self, image_path: str, top_k: int = 10) -> RetrievalResponse:
        """Search similar images by image."""
        start = time.perf_counter()

        query_emb = self._encoder.encode_images([image_path])
        # Request one extra to filter self
        scores, indices = self._indexer.search_images(query_emb, top_k + 1)

        query_filename = Path(image_path).name
        idx_list = [i for i in indices[0].tolist() if i >= 0]
        records = self._metadata.get_images_by_indices(idx_list)
        idx_to_record = {idx: rec for idx, rec in zip(idx_list, records)}

        results = []
        for faiss_idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if faiss_idx < 0 or faiss_idx not in idx_to_record:
                continue
            rec = idx_to_record[faiss_idx]
            if rec.filename == query_filename:
                continue
            results.append(
                RetrievalResult(
                    image_id=rec.image_id,
                    filename=rec.filename,
                    filepath=rec.filepath,
                    score=float(score),
                    captions=rec.captions,
                )
            )
            if len(results) >= top_k:
                break

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResponse(
            query_type="image_to_image",
            results=results,
            total_indexed=self._indexer.image_index_size,
            elapsed_ms=round(elapsed, 2),
        )

    def image_to_text(self, image_path: str, top_k: int = 10) -> RetrievalResponse:
        """Search captions by image."""
        start = time.perf_counter()

        query_emb = self._encoder.encode_images([image_path])
        scores, indices = self._indexer.search_texts(query_emb, top_k)

        caption_indices = [i for i in indices[0].tolist() if i >= 0]
        caption_rows = self._metadata.get_captions_by_indices(caption_indices)
        idx_to_row = {r[0]: r for r in caption_rows}

        results = []
        for faiss_idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if faiss_idx < 0 or faiss_idx not in idx_to_row:
                continue
            _, caption, caption_number, image_id, filename = idx_to_row[faiss_idx]
            results.append(
                CaptionResult(
                    image_id=image_id,
                    filename=filename,
                    caption=caption,
                    caption_number=caption_number,
                    score=float(score),
                )
            )

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResponse(
            query_type="image_to_text",
            results=results,
            total_indexed=self._indexer.text_index_size,
            elapsed_ms=round(elapsed, 2),
        )

    def rag_query(
        self, query: str, top_k: int, generator: LLMGenerator
    ) -> RAGResponse:
        """Retrieve relevant images and generate an LLM answer."""
        retrieval_start = time.perf_counter()
        retrieval_resp = self.text_to_image(query, top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        sources = retrieval_resp.results

        gen_start = time.perf_counter()
        answer = generator.generate(query, sources)
        generation_ms = (time.perf_counter() - gen_start) * 1000

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
        )
