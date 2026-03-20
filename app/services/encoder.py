import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.core.config import settings

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """CLIP encoding service with lazy model loading."""

    def __init__(
        self,
        model_path: str = settings.clip_model_path,
        device: str = settings.device,
        batch_size: int = settings.batch_size,
    ):
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading CLIP model from %s", self._model_path)
        self._processor = CLIPProcessor.from_pretrained(self._model_path)
        self._model = CLIPModel.from_pretrained(self._model_path).to(self._device)
        self._model.eval()
        logger.info("CLIP model loaded on %s", self._device)

    @property
    def dim(self) -> int:
        return 512

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts to L2-normalized embeddings. Returns (N, 512)."""
        self._load_model()
        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                text_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
                text_out = self._model.text_model(**text_inputs)
                embeddings = self._model.text_projection(text_out.pooler_output)

            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        result = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return result / norms

    def encode_images(self, image_paths: list[str | Path]) -> np.ndarray:
        """Encode images to L2-normalized embeddings. Returns (N, 512)."""
        self._load_model()
        all_embeddings = []

        for i in range(0, len(image_paths), self._batch_size):
            batch_paths = image_paths[i : i + self._batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                vision_out = self._model.vision_model(pixel_values=inputs["pixel_values"])
                embeddings = self._model.visual_projection(vision_out.pooler_output)

            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

            # Close images to free memory
            for img in images:
                img.close()

        result = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return result / norms
