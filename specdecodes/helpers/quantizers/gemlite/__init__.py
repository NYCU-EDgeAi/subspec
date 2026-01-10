from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict


def _patch_model_with_gemlite_processor(
    model: Any,
    *,
    device: str,
    processor_cls: Any,
    compute_dtype: Any,
    skip_modules: list[str],
) -> None:
    """Patch `torch.nn.Linear` (and optionally HQQLinear) modules using GemLite processors.

    This is a small, local replacement for `gemlite.helper.patch_model`, which can be
    buggy across gemlite versions.
    """

    import torch

    try:
        from hqq.core.quantize import HQQLinear  # type: ignore
    except Exception:
        HQQLinear = None

    for name, module in model.named_modules():
        setattr(module, "name", name)

    def _make_processor():
        try:
            return processor_cls(device=device, dtype=compute_dtype)
        except TypeError:
            return processor_cls(device=device)

    def _patching_fct(layer: Any):
        layer = layer.to(device, non_blocking=True)

        layer_name = getattr(layer, "name", "")
        if any(s in layer_name for s in skip_modules):
            return layer

        if isinstance(layer, torch.nn.Linear):
            return _make_processor().from_linear(layer)

        if HQQLinear is not None and isinstance(layer, HQQLinear):
            proc = _make_processor()
            if not hasattr(proc, "from_hqqlinear"):
                raise RuntimeError(
                    f"GemLite processor {processor_cls} does not support from_hqqlinear()"
                )
            return proc.from_hqqlinear(layer)

        return layer

    def _patch_linearlayers(mod: Any):
        for name, layer in mod.named_children():
            is_linear = isinstance(layer, torch.nn.Linear)
            is_hqq = HQQLinear is not None and isinstance(layer, HQQLinear)
            if is_linear or is_hqq:
                setattr(mod, name, _patching_fct(layer))
            else:
                _patch_linearlayers(layer)

    _patch_linearlayers(model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class GemliteQuantizer:
    @classmethod
    def quantize_model(cls, model: Any, quant_config: Dict[str, Any], compute_dtype: Any, device: str):
        """Patch a model's Linear layers using GemLite helper processors.

        This is intended for *inference-time* acceleration of the draft model.

        quant_config supports:
          - processor: str, e.g. "A4W4_NVFP_dynamic" (default)
          - skip_modules: list[str], module-name substrings to skip

        Notes:
          - GemLite processors may quantize weights and/or activations depending on processor.
          - On RTX 5090, GemLite ships tuned configs and may auto-load them.
        """

        processor_name = str(quant_config.get("processor") or "A4W4_NVFP_dynamic")
        skip_modules = quant_config.get("skip_modules")
        if skip_modules is None:
            skip_modules = []
        if not isinstance(skip_modules, list):
            raise TypeError("quant_config['skip_modules'] must be a list if provided")
        skip_modules = [str(x) for x in skip_modules]

        try:
            import gemlite  # noqa: F401
            import gemlite.helper as gh
        except Exception as e:
            raise RuntimeError(f"GemLite is required for processor '{processor_name}' but could not be imported: {e}")

        processor_cls = getattr(gh, processor_name, None)
        if processor_cls is None:
            raise ValueError(
                f"Unknown GemLite processor '{processor_name}'. Available processors include: "
                f"{', '.join(sorted([n for n in dir(gh) if n.startswith(('A4','A8','A16'))])[:40])} ..."
            )

        logging.info(f"Patching model with GemLite processor: {processor_name} (skip={skip_modules})")
        _patch_model_with_gemlite_processor(
            model,
            device=device,
            processor_cls=processor_cls,
            compute_dtype=compute_dtype,
            skip_modules=skip_modules,
        )
