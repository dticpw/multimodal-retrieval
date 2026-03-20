from pydantic import BaseModel, Field


class ImageRecord(BaseModel):
    model_config = {"frozen": True}

    image_id: str
    filename: str
    filepath: str
    captions: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    model_config = {"frozen": True}

    image_id: str
    filename: str
    filepath: str
    score: float
    captions: list[str] = Field(default_factory=list)


class CaptionResult(BaseModel):
    model_config = {"frozen": True}

    image_id: str
    filename: str
    caption: str
    caption_number: int
    score: float


class RetrievalResponse(BaseModel):
    model_config = {"frozen": True}

    query_type: str
    results: list[RetrievalResult] | list[CaptionResult]
    total_indexed: int
    elapsed_ms: float


class TextQuery(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class ImageQuery(BaseModel):
    image_path: str
    top_k: int = Field(default=10, ge=1, le=100)


class RAGQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class RAGResponse(BaseModel):
    model_config = {"frozen": True}

    answer: str
    sources: list[RetrievalResult]
    retrieval_ms: float
    generation_ms: float


class EvalMetrics(BaseModel):
    model_config = {"frozen": True}

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
