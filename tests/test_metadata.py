import pytest

from app.services.metadata import MetadataStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = MetadataStore(db_path=db_path)
    return s


class TestMetadataStore:
    def test_insert_and_count(self, store):
        store.insert_image(
            idx=0,
            image_id="img001",
            filename="img001.jpg",
            filepath="/data/img001.jpg",
            captions=[
                (0, 0, "A dog playing"),
                (1, 1, "Dog in the park"),
            ],
        )
        assert store.count_images() == 1
        assert store.count_captions() == 2

    def test_get_images_by_indices(self, store):
        store.insert_image(
            idx=0,
            image_id="img001",
            filename="img001.jpg",
            filepath="/data/img001.jpg",
            captions=[(0, 0, "caption A"), (1, 1, "caption B")],
        )
        store.insert_image(
            idx=1,
            image_id="img002",
            filename="img002.jpg",
            filepath="/data/img002.jpg",
            captions=[(2, 0, "caption C")],
        )

        results = store.get_images_by_indices([0, 1])
        assert len(results) == 2
        assert results[0].image_id == "img001"
        assert results[0].captions == ["caption A", "caption B"]
        assert results[1].image_id == "img002"
        assert results[1].captions == ["caption C"]

    def test_get_images_preserves_order(self, store):
        for i in range(5):
            store.insert_image(
                idx=i,
                image_id=f"img{i:03d}",
                filename=f"img{i:03d}.jpg",
                filepath=f"/data/img{i:03d}.jpg",
                captions=[(i, 0, f"caption {i}")],
            )

        results = store.get_images_by_indices([3, 1, 4])
        assert [r.image_id for r in results] == ["img003", "img001", "img004"]

    def test_get_images_empty(self, store):
        results = store.get_images_by_indices([])
        assert results == []

    def test_get_images_missing_index(self, store):
        store.insert_image(
            idx=0,
            image_id="img001",
            filename="img001.jpg",
            filepath="/data/img001.jpg",
            captions=[],
        )
        results = store.get_images_by_indices([0, 99])
        assert len(results) == 1

    def test_batch_insert(self, store):
        images = [
            (0, "img001", "img001.jpg", "/data/img001.jpg"),
            (1, "img002", "img002.jpg", "/data/img002.jpg"),
        ]
        captions = [
            (0, 0, 0, "cap A"),
            (1, 0, 1, "cap B"),
            (2, 1, 0, "cap C"),
        ]
        store.insert_batch(images, captions)
        assert store.count_images() == 2
        assert store.count_captions() == 3

    def test_get_caption_by_index(self, store):
        store.insert_image(
            idx=0,
            image_id="img001",
            filename="img001.jpg",
            filepath="/data/img001.jpg",
            captions=[(5, 0, "hello world")],
        )
        result = store.get_caption_by_index(5)
        assert result is not None
        caption, caption_num, image_id, filename = result
        assert caption == "hello world"
        assert image_id == "img001"

    def test_get_captions_by_indices(self, store):
        store.insert_image(
            idx=0,
            image_id="img001",
            filename="img001.jpg",
            filepath="/data/img001.jpg",
            captions=[(0, 0, "cap A"), (1, 1, "cap B")],
        )
        rows = store.get_captions_by_indices([0, 1])
        assert len(rows) == 2

    def test_clear(self, store):
        store.insert_image(
            idx=0,
            image_id="img001",
            filename="img001.jpg",
            filepath="/data/img001.jpg",
            captions=[(0, 0, "cap")],
        )
        assert store.count_images() == 1
        store.clear()
        assert store.count_images() == 0
        assert store.count_captions() == 0
