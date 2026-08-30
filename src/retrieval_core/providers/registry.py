from typing import Protocol

from retrieval_core.providers.base import ModelProvider

_REGISTRY: dict[str, ModelProvider] = {}

class ProviderRegistry(Protocol):
    def register(self, name: str, provider: ModelProvider) -> None:
        ...

    def get(self, name: str) -> ModelProvider:
        ...

def register_provider(name: str, provider: ModelProvider) -> None:
    _REGISTRY[name] = provider

def get_provider(name: str) -> ModelProvider:
    if name not in _REGISTRY:
        raise ValueError(f"Provider {name} not found")
    return _REGISTRY[name]

def clear_registry() -> None:
    _REGISTRY.clear()
