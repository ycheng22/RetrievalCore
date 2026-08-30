import asyncio

from core.models import Query, ScoreBreakdown, ScoredHit
from core.retrievers.base import Retriever


class DummyRetriever:
    async def retrieve(self, query: Query, top_k: int) -> list[ScoredHit]:
        return [
            ScoredHit(
                product_id="test_id",
                raw_score=1.0,
                retriever_name="dummy",
                rank=1,
                breakdown=ScoreBreakdown()
            )
        ]

def test_dummy_retriever_protocol() -> None:
    retriever: Retriever = DummyRetriever()
    query = Query(text="test query")
    hits = asyncio.run(retriever.retrieve(query, top_k=5))
    assert len(hits) == 1
    assert hits[0].product_id == "test_id"
