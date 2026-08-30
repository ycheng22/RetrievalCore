from typing import Protocol

from core.models import Query, ScoredHit


class Retriever(Protocol):
    async def retrieve(self, query: Query, top_k: int) -> list[ScoredHit]:
        ...
