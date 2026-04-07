"""Data ingestion CLI: parse Flickr30K CSV, encode with CLIP, build FAISS indexes."""

import argparse
import csv
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.services.encoder import CLIPEncoder
from app.services.indexer import FAISSIndexer
from app.services.metadata import MetadataStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_flickr_csv(csv_path: str) -> dict[str, list[tuple[int, str]]]:
    """Parse Flickr30K CSV. Returns {filename: [(comment_number, caption), ...]}."""
    data: dict[str, list[tuple[int, str]]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)  # skip header
        logger.info("CSV header: %s", header)

        for row in reader:
            if len(row) < 3:
                continue
            filename = row[0].strip()
            try:
                comment_num = int(row[1].strip())
            except ValueError:
                continue
            caption = row[2].strip()
            if filename and caption:
                data[filename].append((comment_num, caption))

    logger.info("Parsed %d images from %s", len(data), csv_path)
    return dict(data)


def run_ingestion(csv_path: str, image_dir: str) -> None:
    """Full ingestion pipeline."""
    start_time = time.time()

    # 1. Parse CSV
    logger.info("Parsing CSV: %s", csv_path)
    image_data = parse_flickr_csv(csv_path)
    image_names = sorted(image_data.keys())
    logger.info("Total images: %d", len(image_names))

    # 2. Prepare image paths and verify existence
    image_dir_path = Path(image_dir)
    valid_images = []
    for name in image_names:
        path = image_dir_path / name
        if path.exists():
            valid_images.append((name, str(path)))
        else:
            logger.warning("Image not found: %s", path)

    logger.info("Valid images: %d / %d", len(valid_images), len(image_names))

    # 3. Encode images in batches
    encoder = CLIPEncoder()
    image_paths = [p for _, p in valid_images]
    logger.info("Encoding %d images in batches of %d...", len(image_paths), settings.batch_size)

    image_emb_list = []
    for i in tqdm(range(0, len(image_paths), settings.batch_size), desc="Encoding images"):
        batch = image_paths[i:i+settings.batch_size]
        emb = encoder.encode_images(batch)
        image_emb_list.append(emb)
    image_embeddings = np.vstack(image_emb_list)
    logger.info("Image embeddings shape: %s", image_embeddings.shape)

    # 4. Encode captions and build caption mapping
    all_captions = []  # (caption_idx, image_idx, caption_number, caption_text)
    all_caption_texts = []
    caption_idx = 0

    for image_idx, (name, _) in enumerate(valid_images):
        for comment_num, caption in image_data[name]:
            all_captions.append((caption_idx, image_idx, comment_num, caption))
            all_caption_texts.append(caption)
            caption_idx += 1

    logger.info("Encoding %d captions in batches of %d...", len(all_caption_texts), settings.batch_size)
    text_emb_list = []
    for i in tqdm(range(0, len(all_caption_texts), settings.batch_size), desc="Encoding captions"):
        batch = all_caption_texts[i:i+settings.batch_size]
        emb = encoder.encode_texts(batch)
        text_emb_list.append(emb)
    text_embeddings = np.vstack(text_emb_list)
    logger.info("Text embeddings shape: %s", text_embeddings.shape)

    # 5. Build FAISS indexes
    indexer = FAISSIndexer()
    logger.info("Building indexes (IVF mode: %s)...", settings.use_ivf_index)
    indexer.build_image_index(image_embeddings)
    indexer.build_text_index(text_embeddings)
    indexer.save()
    logger.info(
        "FAISS indexes saved: %d images, %d captions",
        indexer.image_index_size,
        indexer.text_index_size,
    )

    # 6. Store metadata
    metadata = MetadataStore()
    metadata.clear()

    images_batch = []
    captions_batch = []
    for image_idx, (name, path) in enumerate(valid_images):
        image_id = Path(name).stem
        images_batch.append((image_idx, image_id, name, path))

    for caption_idx, image_idx, caption_number, caption_text in all_captions:
        captions_batch.append((caption_idx, image_idx, caption_number, caption_text))

    metadata.insert_batch(images_batch, captions_batch)
    logger.info(
        "Metadata stored: %d images, %d captions",
        metadata.count_images(),
        metadata.count_captions(),
    )
    metadata.close()

    elapsed = time.time() - start_time
    logger.info("Ingestion complete in %.1f seconds", elapsed)


def main():
    parser = argparse.ArgumentParser(description="Ingest Flickr30K data")
    parser.add_argument(
        "--csv",
        default=settings.flickr30k_train4k_csv,
        help="CSV file path (default: train4K.csv)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full train.csv instead of 4K subset",
    )
    parser.add_argument(
        "--image-dir",
        default=settings.flickr30k_image_dir,
        help="Image directory path",
    )
    args = parser.parse_args()

    csv_path = settings.flickr30k_train_csv if args.full else args.csv
    run_ingestion(csv_path, args.image_dir)


if __name__ == "__main__":
    main()
