from typing import Protocol


class ModelProvider(Protocol):
    async def generate(self, prompt: str) -> str:
        ...

    async def embed(self, texts: list[str], dim: int) -> list[list[float]]:
        ...
