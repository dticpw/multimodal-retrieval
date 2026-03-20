import numpy as np
import pytest

from app.services.encoder import CLIPEncoder


@pytest.fixture(scope="module")
def encoder():
    return CLIPEncoder(device="cpu")


class TestCLIPEncoder:
    def test_encode_texts_shape(self, encoder):
        texts = ["a dog", "a cat", "a bird"]
        emb = encoder.encode_texts(texts)
        assert emb.shape == (3, 512)
        assert emb.dtype == np.float32

    def test_encode_texts_normalized(self, encoder):
        texts = ["hello world"]
        emb = encoder.encode_texts(texts)
        norm = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_encode_texts_semantic_similarity(self, encoder):
        texts = ["a photo of a dog", "a picture of a puppy", "the stock market is down"]
        emb = encoder.encode_texts(texts)
        # dog and puppy should be more similar than dog and stock market
        sim_dog_puppy = emb[0] @ emb[1]
        sim_dog_stock = emb[0] @ emb[2]
        assert sim_dog_puppy > sim_dog_stock

    def test_encode_images_shape(self, encoder, tmp_path):
        from PIL import Image

        # Create dummy images
        paths = []
        for i in range(2):
            img = Image.new("RGB", (224, 224), color=(i * 50, 100, 150))
            p = tmp_path / f"test_{i}.jpg"
            img.save(str(p))
            paths.append(str(p))

        emb = encoder.encode_images(paths)
        assert emb.shape == (2, 512)
        assert emb.dtype == np.float32

    def test_encode_images_normalized(self, encoder, tmp_path):
        from PIL import Image

        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        p = tmp_path / "test.jpg"
        img.save(str(p))

        emb = encoder.encode_images([str(p)])
        norm = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_dim_property(self, encoder):
        assert encoder.dim == 512
