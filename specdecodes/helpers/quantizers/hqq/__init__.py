import logging
import importlib.util
import os

from hqq.core.quantize import *
from hqq.core.peft import HQQLinearLoRA
from hqq.utils.patching import prepare_for_inference
from .hf.base import AutoHQQHFModel

# Check for the kernel library (replace 'kernel_library' with the actual module name)
if importlib.util.find_spec("gemlite") is None:
    raise ImportError(
        "The 'gemlite' kernel is required for optimized performance. "
        "Please install it using 'pip install git+https://github.com/mobiusml/gemlite/'."
    )

class HqqQuantizer:
    @classmethod
    def quantize_model(cls, model, quant_config, compute_dtype, device):
        logging.info("Quantizing model with HqqQuantizer")

        # Optional GemLite tuning knobs (no-op unless env vars are set).
        # These primarily reduce warmup/autotune overhead and can improve steady-state
        # performance depending on GPU + shapes.
        gemlite_cfg = os.environ.get("SUBSPEC_GEMLITE_CONFIG") or os.environ.get("GEMLITE_CONFIG")
        gemlite_autotune = os.environ.get("SUBSPEC_GEMLITE_AUTOTUNE") or os.environ.get("GEMLITE_AUTOTUNE")
        gemlite_packing = os.environ.get("SUBSPEC_GEMLITE_PACKING_BITWIDTH")
        gemlite_kernel_caching = os.environ.get("SUBSPEC_GEMLITE_KERNEL_CACHING")
        if any([gemlite_cfg, gemlite_autotune, gemlite_packing, gemlite_kernel_caching]):
            try:
                import gemlite  # type: ignore

                if gemlite_autotune:
                    try:
                        gemlite.set_autotune(str(gemlite_autotune))
                        logging.info(f"GemLite autotune mode set to: {gemlite_autotune}")
                    except Exception as e:
                        logging.warning(f"Failed to set GemLite autotune mode '{gemlite_autotune}': {e}")

                if gemlite_packing:
                    try:
                        gemlite.set_packing_bitwidth(int(gemlite_packing))
                        logging.info(f"GemLite packing bitwidth set to: {gemlite_packing}")
                    except Exception as e:
                        logging.warning(f"Failed to set GemLite packing bitwidth '{gemlite_packing}': {e}")

                if gemlite_kernel_caching is not None:
                    try:
                        val = str(gemlite_kernel_caching).strip().lower()
                        enabled = val in {"1", "true", "yes", "y", "on"}
                        gemlite.set_kernel_caching(enabled)
                        logging.info(f"GemLite kernel caching set to: {enabled}")
                    except Exception as e:
                        logging.warning(f"Failed to set GemLite kernel caching: {e}")

                if gemlite_cfg:
                    try:
                        gemlite.load_config(str(gemlite_cfg))
                        logging.info(f"Loaded GemLite config: {gemlite_cfg}")
                    except Exception as e:
                        logging.warning(f"Failed to load GemLite config '{gemlite_cfg}': {e}")
            except Exception as e:
                logging.warning(f"GemLite tuning requested but gemlite import/config failed: {e}")

        AutoHQQHFModel.quantize_model(model, quant_config, compute_dtype=compute_dtype, device=device)
        HQQLinear.set_backend(HQQBackend.PYTORCH)

        # By default, HQQ's gemlite backend patches HQQLinear -> GemLite A16Wn (FP16 activations, Wn weights).
        # GemLite activation quant (e.g. A8Wn_dynamic) is not wired through HQQ's backend in gemlite==0.4.6,
        # so we provide an opt-in path here.
        #
        # Supported values:
        # - SUBSPEC_GEMLITE_ACTIVATIONS=fp16 (default): weight-only, uses prepare_for_inference(..., backend="gemlite")
        # - SUBSPEC_GEMLITE_ACTIVATIONS=fp8: FP8 dynamic activations + Wn weights via gemlite.helper.A8Wn_dynamic
        act_mode = (os.environ.get("SUBSPEC_GEMLITE_ACTIVATIONS") or "fp16").strip().lower()
        if act_mode in {"fp8", "a8", "a8wn"}:
            try:
                from gemlite.helper import A8Wn_dynamic

                use_fp8e5 = (os.environ.get("SUBSPEC_GEMLITE_FP8_FORMAT") or "e4m3").strip().lower() in {
                    "e5m2",
                    "fp8e5",
                    "e5",
                }
                post_scale = str(os.environ.get("SUBSPEC_GEMLITE_POST_SCALE") or "0").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                }

                processor = A8Wn_dynamic(device=device, post_scale=post_scale, use_fp8e5=use_fp8e5)

                def _patch_module(mod):
                    for child_name, child in list(mod._modules.items()):
                        if child is None:
                            continue

                        if isinstance(child, HQQLinear):
                            mod._modules[child_name] = processor.from_hqqlinear(child)
                            continue

                        if isinstance(child, HQQLinearLoRA):
                            # Keep LoRA wrapper, replace underlying quantized linear.
                            if isinstance(child.linear_layer, HQQLinear):
                                child.linear_layer = processor.from_hqqlinear(child.linear_layer)
                            continue

                        _patch_module(child)

                _patch_module(model)
                logging.info("Patched HQQLinear layers with GemLite A8Wn_dynamic (FP8 dynamic activations).")
                return
            except Exception as e:
                logging.warning(
                    f"SUBSPEC_GEMLITE_ACTIVATIONS={act_mode} requested but GemLite FP8 patching failed; falling back to A16Wn weight-only. Error: {e}"
                )

        prepare_for_inference(model, backend="gemlite")