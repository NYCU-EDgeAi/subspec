from __future__ import annotations

import importlib
from typing import Any, Dict


def load_symbol(path: str):
    """Load a symbol from a module path.

    Supports:
    - "pkg.mod:ClassName"
    - "pkg.mod.ClassName"
    """
    if ":" in path:
        module_name, symbol_name = path.split(":", 1)
    else:
        module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def instantiate_recipe(recipe_spec: Any):
    """Normalize config.recipe.

    - If recipe_spec is already an object instance (not dict/str), return as-is.
    - If recipe_spec is a mapping, instantiate from class_path/module+class and kwargs.
    - If recipe_spec is a string, treat it as class_path.

    YAML example:
        recipe:
          class_path: specdecodes.helpers.recipes.offload.layer_offload:LayerOffloadRecipe
          kwargs:
            keep_first_n_layers_on_gpu: 7
    """
    if recipe_spec is None:
        return None

    if not isinstance(recipe_spec, (dict, str)):
        return recipe_spec

    kwargs: Dict[str, Any] = {}
    if isinstance(recipe_spec, dict):
        class_path = recipe_spec.get("class_path") or recipe_spec.get("path")
        if not class_path and isinstance(recipe_spec.get("module"), str):
            module = recipe_spec["module"]
            cls_name = recipe_spec.get("class", "Recipe")
            class_path = f"{module}:{cls_name}"

        if not isinstance(class_path, str) or not class_path.strip():
            raise ValueError(
                "recipe must specify 'class_path' (or 'module' + 'class') when provided as a mapping"
            )

        kwargs_val = recipe_spec.get("kwargs")
        if isinstance(kwargs_val, dict):
            kwargs = dict(kwargs_val)
    else:
        class_path = recipe_spec

    cls = load_symbol(class_path)
    return cls(**kwargs) if kwargs else cls()
