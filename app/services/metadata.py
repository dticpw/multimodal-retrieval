import logging
import sqlite3
from pathlib import Path

from app.core.config import settings
from app.models.schemas import ImageRecord

logger = logging.getLogger(__name__)


class MetadataStore:
    """SQLite-backed metadata storage for images and captions."""

    def __init__(self, db_path: Path | str = settings.metadata_db_path):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_tables()
        return self._conn

    def _create_tables(self) -> None:
        conn = self._conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                idx INTEGER PRIMARY KEY,
                image_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS captions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caption_idx INTEGER NOT NULL,
                image_idx INTEGER NOT NULL,
                caption_number INTEGER NOT NULL,
                caption TEXT NOT NULL,
                FOREIGN KEY (image_idx) REFERENCES images(idx)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_captions_image ON captions(image_idx)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_captions_cidx ON captions(caption_idx)"
        )
        conn.commit()

    def clear(self) -> None:
        """Delete all data."""
        conn = self._connect()
        conn.execute("DELETE FROM captions")
        conn.execute("DELETE FROM images")
        conn.commit()

    def insert_image(
        self,
        idx: int,
        image_id: str,
        filename: str,
        filepath: str,
        captions: list[tuple[int, int, str]],
    ) -> None:
        """Insert an image and its captions.

        Args:
            idx: FAISS index position
            image_id: unique image identifier
            filename: image filename
            filepath: full path to image
            captions: list of (caption_idx, caption_number, caption_text)
        """
        conn = self._connect()
        conn.execute(
            "INSERT INTO images (idx, image_id, filename, filepath) VALUES (?, ?, ?, ?)",
            (idx, image_id, filename, filepath),
        )
        for caption_idx, caption_number, caption_text in captions:
            conn.execute(
                "INSERT INTO captions (caption_idx, image_idx, caption_number, caption) "
                "VALUES (?, ?, ?, ?)",
                (caption_idx, idx, caption_number, caption_text),
            )
        conn.commit()

    def insert_batch(
        self,
        images: list[tuple[int, str, str, str]],
        captions: list[tuple[int, int, int, str]],
    ) -> None:
        """Batch insert images and captions.

        Args:
            images: list of (idx, image_id, filename, filepath)
            captions: list of (caption_idx, image_idx, caption_number, caption_text)
        """
        conn = self._connect()
        conn.executemany(
            "INSERT INTO images (idx, image_id, filename, filepath) VALUES (?, ?, ?, ?)",
            images,
        )
        conn.executemany(
            "INSERT INTO captions (caption_idx, image_idx, caption_number, caption) "
            "VALUES (?, ?, ?, ?)",
            captions,
        )
        conn.commit()

    def get_images_by_indices(self, indices: list[int]) -> list[ImageRecord]:
        """Retrieve images with captions by FAISS indices."""
        conn = self._connect()
        if not indices:
            return []

        placeholders = ",".join("?" for _ in indices)
        rows = conn.execute(
            f"SELECT idx, image_id, filename, filepath FROM images WHERE idx IN ({placeholders})",
            indices,
        ).fetchall()

        idx_to_row = {r[0]: r for r in rows}

        results = []
        for idx in indices:
            if idx not in idx_to_row:
                continue
            _, image_id, filename, filepath = idx_to_row[idx]
            caption_rows = conn.execute(
                "SELECT caption FROM captions WHERE image_idx = ? ORDER BY caption_number",
                (idx,),
            ).fetchall()
            captions = [c[0] for c in caption_rows]
            results.append(
                ImageRecord(
                    image_id=image_id,
                    filename=filename,
                    filepath=filepath,
                    captions=captions,
                )
            )
        return results

    def get_caption_by_index(self, caption_idx: int) -> tuple[str, int, str, str] | None:
        """Get caption info by caption_idx. Returns (caption, caption_number, image_id, filename)."""
        conn = self._connect()
        row = conn.execute(
            """
            SELECT c.caption, c.caption_number, i.image_id, i.filename
            FROM captions c JOIN images i ON c.image_idx = i.idx
            WHERE c.caption_idx = ?
            """,
            (caption_idx,),
        ).fetchone()
        return row

    def get_captions_by_indices(
        self, caption_indices: list[int]
    ) -> list[tuple[int, str, int, str, str]]:
        """Get multiple captions by caption_idx.
        Returns list of (caption_idx, caption, caption_number, image_id, filename).
        """
        conn = self._connect()
        if not caption_indices:
            return []
        placeholders = ",".join("?" for _ in caption_indices)
        rows = conn.execute(
            f"""
            SELECT c.caption_idx, c.caption, c.caption_number, i.image_id, i.filename
            FROM captions c JOIN images i ON c.image_idx = i.idx
            WHERE c.caption_idx IN ({placeholders})
            """,
            caption_indices,
        ).fetchall()
        return rows

    def count_images(self) -> int:
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def count_captions(self) -> int:
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
