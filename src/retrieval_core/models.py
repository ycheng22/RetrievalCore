
from typing import Any
from pydantic import BaseModel, Field


class Query(BaseModel):
    text: str

class Product(BaseModel):
    product_id: str
    title: str

class Document(BaseModel):
    doc_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class ScoreBreakdown(BaseModel):
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    rank_before_rerank: int | None = None
    matched_terms: list[str] = Field(default_factory=list)

class ScoredHit(BaseModel):
    product_id: str | None = None
    doc_id: str | None = None
    raw_score: float
    retriever_name: str
    rank: int
    breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)

class SearchResponse(BaseModel):
    hits: list[ScoredHit]
    total_found: int

class PipelineConfig(BaseModel):
    use_bm25: bool
    use_dense: bool
    fusion_method: str
    use_rerank: bool
    embed_dim: int
    top_k: int
