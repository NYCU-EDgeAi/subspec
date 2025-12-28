"""Offloading recipes.

This package contains offloading-only recipes (no quantization) that can be
instantiated from YAML via `recipe.class_path`.

Recommended:
- LayerOffloadRecipe: keep first N layers on GPU, offload the rest to CPU.
"""

from .layer_offload import LayerOffloadRecipe, Recipe

__all__ = ["LayerOffloadRecipe", "Recipe"]
