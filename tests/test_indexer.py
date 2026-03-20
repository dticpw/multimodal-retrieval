import numpy as np
import pytest

from app.services.indexer import FAISSIndexer


def _random_normalized(n: int, dim: int = 512) -> np.ndarray:
    x = np.random.randn(n, dim).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


class TestFAISSIndexer:
    def test_build_image_index(self):
        indexer = FAISSIndexer(dim=512)
        emb = _random_normalized(100)
        indexer.build_image_index(emb)
        assert indexer.image_index_size == 100

    def test_build_text_index(self):
        indexer = FAISSIndexer(dim=512)
        emb = _random_normalized(500)
        indexer.build_text_index(emb)
        assert indexer.text_index_size == 500

    def test_search_images(self):
        indexer = FAISSIndexer(dim=512)
        emb = _random_normalized(50)
        indexer.build_image_index(emb)

        query = emb[0:1]  # query with first vector
        scores, indices = indexer.search_images(query, top_k=5)

        assert scores.shape == (1, 5)
        assert indices.shape == (1, 5)
        # First result should be itself (exact match)
        assert indices[0, 0] == 0
        np.testing.assert_allclose(scores[0, 0], 1.0, atol=1e-5)

    def test_search_texts(self):
        indexer = FAISSIndexer(dim=512)
        emb = _random_normalized(200)
        indexer.build_text_index(emb)

        query = emb[10:11]
        scores, indices = indexer.search_texts(query, top_k=3)

        assert indices[0, 0] == 10
        np.testing.assert_allclose(scores[0, 0], 1.0, atol=1e-5)

    def test_search_1d_query(self):
        indexer = FAISSIndexer(dim=512)
        emb = _random_normalized(10)
        indexer.build_image_index(emb)

        query = emb[0]  # 1D
        scores, indices = indexer.search_images(query, top_k=3)
        assert scores.shape == (1, 3)

    def test_save_load(self, tmp_path):
        indexer = FAISSIndexer(dim=512, index_dir=tmp_path)
        img_emb = _random_normalized(20)
        txt_emb = _random_normalized(100)
        indexer.build_image_index(img_emb)
        indexer.build_text_index(txt_emb)
        indexer.save()

        # Load into new indexer
        indexer2 = FAISSIndexer(dim=512, index_dir=tmp_path)
        indexer2.load()
        assert indexer2.image_index_size == 20
        assert indexer2.text_index_size == 100

        # Search should give same results
        query = img_emb[0:1]
        s1, i1 = indexer.search_images(query, top_k=5)
        s2, i2 = indexer2.search_images(query, top_k=5)
        np.testing.assert_array_equal(i1, i2)

    def test_search_without_index_raises(self):
        indexer = FAISSIndexer(dim=512)
        query = _random_normalized(1)
        with pytest.raises(RuntimeError):
            indexer.search_images(query)

    def test_load_missing_raises(self, tmp_path):
        indexer = FAISSIndexer(dim=512, index_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            indexer.load()
