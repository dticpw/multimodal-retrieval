from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # CLIP model
    clip_model_path: str = "E:/PG/model/clip"

    # Flickr30K paths
    flickr30k_image_dir: str = "E:/PG/dataset/flickr_30k/flickr30k-images"
    flickr30k_train_csv: str = "E:/PG/dataset/flickr_30k/train.csv"
    flickr30k_train4k_csv: str = "E:/PG/dataset/flickr_30k/train4K.csv"
    flickr30k_test_csv: str = "E:/PG/dataset/flickr_30k/test.csv"

    # Storage
    index_dir: str = "indexes"
    data_dir: str = "data"

    # Encoding
    batch_size: int = 64
    device: str = "cuda"

    # FAISS indexing
    use_ivf_index: bool = False  # True for >50K vectors
    ivf_nlist: int = 100  # Number of clusters (sqrt(N) recommended)
    ivf_nprobe: int = 10  # Search probes (higher = more accurate but slower)

    # LLM
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def metadata_db_path(self) -> Path:
        return self.data_path / "metadata.db"


settings = Settings()
