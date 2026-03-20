"""Flickr30K standard evaluation: Recall@1/5/10 for text→image and image→text."""

import csv
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.models.schemas import EvalMetrics
from app.services.encoder import CLIPEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_test_csv(csv_path: str) -> dict[str, list[str]]:
    """Parse test CSV. Returns {filename: [caption0, caption1, ...]}."""
    data: dict[str, list[str]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            filename = row[0].strip()
            caption = row[2].strip()
            if filename and caption:
                data[filename].append(caption)
    return dict(data)


def compute_recall(similarity: np.ndarray, ks: list[int] = [1, 5, 10]) -> dict[int, float]:
    """Compute Recall@K from a similarity matrix.

    Args:
        similarity: (N_query, N_gallery) similarity matrix.
                    Ground truth: query i matches gallery i (for t2i)
                    or computed per the specific protocol.
        ks: list of K values.

    Returns:
        {k: recall_at_k}
    """
    n_query = similarity.shape[0]
    # Rank by descending similarity
    ranks = np.argsort(-similarity, axis=1)

    results = {}
    for k in ks:
        # For each query, check if the ground truth index is in top-k
        hits = 0
        for i in range(n_query):
            if i in ranks[i, :k]:
                hits += 1
        results[k] = hits / n_query
    return results


def run_benchmark(test_csv: str = settings.flickr30k_test_csv, image_dir: str = settings.flickr30k_image_dir):
    """Run standard Flickr30K evaluation."""
    start = time.time()

    # Parse test data
    test_data = parse_test_csv(test_csv)
    image_names = sorted(test_data.keys())
    logger.info("Test set: %d images", len(image_names))

    # Verify images exist
    image_dir_path = Path(image_dir)
    valid = [(name, str(image_dir_path / name)) for name in image_names if (image_dir_path / name).exists()]
    logger.info("Valid images: %d", len(valid))

    if len(valid) < len(image_names):
        logger.warning("Missing %d images", len(image_names) - len(valid))

    names = [v[0] for v in valid]
    paths = [v[1] for v in valid]

    # Build caption list (5 per image, ordered)
    all_captions = []
    for name in names:
        caps = test_data[name]
        # Pad or truncate to 5
        caps = (caps + [""] * 5)[:5]
        all_captions.extend(caps)

    n_images = len(names)
    n_captions = len(all_captions)
    logger.info("Images: %d, Captions: %d", n_images, n_captions)

    # Encode
    encoder = CLIPEncoder()

    logger.info("Encoding images...")
    image_emb = encoder.encode_images(paths)  # (N, 512)

    logger.info("Encoding captions...")
    text_emb = encoder.encode_texts(all_captions)  # (5N, 512)

    # Compute similarity matrix: (5N, N) for text→image
    logger.info("Computing similarity matrix...")
    t2i_sim = text_emb @ image_emb.T  # (5N, N)

    # Text→Image Recall: for caption i*5+j, ground truth image is i
    logger.info("=== Text → Image Retrieval ===")
    t2i_ranks = np.argsort(-t2i_sim, axis=1)
    t2i_recall = {}
    for k in [1, 5, 10]:
        hits = 0
        for q_idx in range(n_captions):
            gt_image = q_idx // 5  # caption j belongs to image j//5
            if gt_image in t2i_ranks[q_idx, :k]:
                hits += 1
        t2i_recall[k] = hits / n_captions

    t2i_metrics = EvalMetrics(
        recall_at_1=round(t2i_recall[1] * 100, 2),
        recall_at_5=round(t2i_recall[5] * 100, 2),
        recall_at_10=round(t2i_recall[10] * 100, 2),
    )
    logger.info("Text→Image: R@1=%.2f%%, R@5=%.2f%%, R@10=%.2f%%",
                t2i_metrics.recall_at_1, t2i_metrics.recall_at_5, t2i_metrics.recall_at_10)

    # Image→Text Recall: for image i, ground truth captions are i*5..i*5+4
    logger.info("=== Image → Text Retrieval ===")
    i2t_sim = image_emb @ text_emb.T  # (N, 5N)
    i2t_ranks = np.argsort(-i2t_sim, axis=1)
    i2t_recall = {}
    for k in [1, 5, 10]:
        hits = 0
        for q_idx in range(n_images):
            gt_captions = set(range(q_idx * 5, q_idx * 5 + 5))
            top_k_set = set(i2t_ranks[q_idx, :k].tolist())
            if gt_captions & top_k_set:
                hits += 1
        i2t_recall[k] = hits / n_images

    i2t_metrics = EvalMetrics(
        recall_at_1=round(i2t_recall[1] * 100, 2),
        recall_at_5=round(i2t_recall[5] * 100, 2),
        recall_at_10=round(i2t_recall[10] * 100, 2),
    )
    logger.info("Image→Text: R@1=%.2f%%, R@5=%.2f%%, R@10=%.2f%%",
                i2t_metrics.recall_at_1, i2t_metrics.recall_at_5, i2t_metrics.recall_at_10)

    elapsed = time.time() - start
    logger.info("Benchmark complete in %.1f seconds", elapsed)

    return {"text_to_image": t2i_metrics, "image_to_text": i2t_metrics}


if __name__ == "__main__":
    run_benchmark()
