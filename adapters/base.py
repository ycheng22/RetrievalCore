from typing import Protocol

from core.models import Product


class CorpusAdapter(Protocol):
    async def get_product(self, product_id: str) -> Product | None:
        ...
    
    async def mget_products(self, product_ids: list[str]) -> list[Product]:
        ...
