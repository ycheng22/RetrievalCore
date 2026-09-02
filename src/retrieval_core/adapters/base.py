from typing import Protocol

from retrieval_core.models import Product, Document


class CorpusAdapter(Protocol):
    async def get_product(self, product_id: str) -> Product | None:
        ...
    
    async def mget_products(self, product_ids: list[str]) -> list[Product]:
        ...

class DocumentAdapter(Protocol):
    async def get_document(self, doc_id: str) -> Document | None:
        ...
    
    async def mget_documents(self, doc_ids: list[str]) -> list[Document]:
        ...
