import logging
from pathlib import Path

import faiss
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """Manages dual FAISS indexes: image_index and text_index."""

    def __init__(self, dim: int = 512, index_dir: str | Path = settings.index_dir):
        self._dim = dim
        self._index_dir = Path(index_dir)
        self._image_index: faiss.Index | None = None
        self._text_index: faiss.Index | None = None

    @property
    def image_index_size(self) -> int:
        return self._image_index.ntotal if self._image_index else 0

    @property
    def text_index_size(self) -> int:
        return self._text_index.ntotal if self._text_index else 0

    def build_image_index(self, embeddings: np.ndarray) -> None:
        """Build image index from L2-normalized embeddings. Shape: (N, dim)."""
        assert embeddings.ndim == 2 and embeddings.shape[1] == self._dim
        self._image_index = faiss.IndexFlatIP(self._dim)
        self._image_index.add(embeddings.astype(np.float32))
        logger.info("Built image index with %d vectors", self._image_index.ntotal)

    def build_text_index(self, embeddings: np.ndarray) -> None:
        """Build text index from L2-normalized embeddings. Shape: (M, dim)."""
        assert embeddings.ndim == 2 and embeddings.shape[1] == self._dim
        self._text_index = faiss.IndexFlatIP(self._dim)
        self._text_index.add(embeddings.astype(np.float32))
        logger.info("Built text index with %d vectors", self._text_index.ntotal)

    def search_images(self, query: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search image index. Returns (scores, indices) of shape (N, top_k)."""
        if self._image_index is None:
            raise RuntimeError("Image index not built or loaded")
        query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, indices = self._image_index.search(query, top_k)
        return scores, indices

    def search_texts(self, query: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search text index. Returns (scores, indices) of shape (N, top_k)."""
        if self._text_index is None:
            raise RuntimeError("Text index not built or loaded")
        query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, indices = self._text_index.search(query, top_k)
        return scores, indices

    def save(self) -> None:
        """Save both indexes to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        if self._image_index:
            path = str(self._index_dir / "image.index")
            faiss.write_index(self._image_index, path)
            logger.info("Saved image index to %s", path)
        if self._text_index:
            path = str(self._index_dir / "text.index")
            faiss.write_index(self._text_index, path)
            logger.info("Saved text index to %s", path)

    def load(self) -> None:
        """Load both indexes from disk."""
        image_path = self._index_dir / "image.index"
        text_path = self._index_dir / "text.index"

        if image_path.exists():
            self._image_index = faiss.read_index(str(image_path))
            logger.info("Loaded image index: %d vectors", self._image_index.ntotal)
        else:
            raise FileNotFoundError(f"Image index not found: {image_path}")

        if text_path.exists():
            self._text_index = faiss.read_index(str(text_path))
            logger.info("Loaded text index: %d vectors", self._text_index.ntotal)
        else:
            raise FileNotFoundError(f"Text index not found: {text_path}")
