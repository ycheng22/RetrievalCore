from retrieval_core.models import (
    PipelineConfig,
    ScoreBreakdown,
    ScoredHit,
)


def test_score_breakdown_additive() -> None:
    b = ScoreBreakdown(bm25_score=1.0, dense_score=2.0, fused_score=3.0)
    assert b.bm25_score == 1.0
    assert b.dense_score == 2.0
    assert b.fused_score == 3.0

def test_pipeline_config() -> None:
    config = PipelineConfig(
        use_bm25=True,
        use_dense=True,
        fusion_method="rrf",
        use_rerank=False,
        embed_dim=768,
        top_k=10,
    )
    assert config.embed_dim == 768
    assert config.use_bm25 is True

def test_scored_hit_validation() -> None:
    hit = ScoredHit(
        product_id="P1",
        raw_score=0.9,
        retriever_name="bm25",
        rank=1
    )
    assert hit.product_id == "P1"
    assert hit.breakdown.bm25_score == 0.0
