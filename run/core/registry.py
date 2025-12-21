from typing import Any, Dict, Optional, Callable, Callable
from dataclasses import dataclass

@dataclass
class ModelRegistryEntry:
    name: str
    generator_cls: Any
    draft_model_cls: Any
    default_config: Dict[str, Any]
    load_model_fn: Optional[Callable] = None
    load_draft_model_fn: Optional[Callable] = None
    load_kv_cache_fn: Optional[Callable] = None

class ModelRegistry:
    _registry: Dict[str, ModelRegistryEntry] = {}

    @classmethod
    def register(cls, name: str, generator_cls: Any, draft_model_cls: Any, default_config: Dict[str, Any] = None,
                 load_model_fn: Optional[Callable] = None, load_draft_model_fn: Optional[Callable] = None,
                 load_kv_cache_fn: Optional[Callable] = None):
        if default_config is None:
            default_config = {}
        cls._registry[name] = ModelRegistryEntry(
            name=name,
            generator_cls=generator_cls,
            draft_model_cls=draft_model_cls,
            default_config=default_config,
            load_model_fn=load_model_fn,
            load_draft_model_fn=load_draft_model_fn,
            load_kv_cache_fn=load_kv_cache_fn
        )

    @classmethod
    def get(cls, name: str) -> Optional[ModelRegistryEntry]:
        return cls._registry.get(name)

    @classmethod
    def list_methods(cls):
        return list(cls._registry.keys())
