"""SubSpec recipe: target postspec offloading + draft GemLite patching.

This is the generic version of the earlier NVFP4-named recipe.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..base_recipe import QuantOffloadRecipe
from ...offloaders.prefetch_offloader_postspec import PrefetchOffloader
from ...quantizers.gemlite import GemliteQuantizer

from hqq.core.quantize import BaseQuantizeConfig


class Recipe(QuantOffloadRecipe):
    def __init__(self, processor: str = "A4W4_NVFP_dynamic", skip_modules: Optional[list[str]] = None):
        super().__init__()
        self.quantizer = GemliteQuantizer
        self.offloader = PrefetchOffloader
        self.processor = str(processor)
        self.skip_modules = list(skip_modules) if skip_modules is not None else []

    def generate_configurations(
        self,
        target_model: Any,
        draft_model: Any,
        max_length: int,
        cpu_offload_gb: Optional[int],
        dtype: Any,
        device: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Same device_map logic as HQQ postspec recipe:
        # offload heavy linear weights to CPU for the target model.
        quant_config: Dict[str, Any] = {}
        attn_quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
        mlp_quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)

        layer_cnt = len(target_model.model.layers)
        for i in range(layer_cnt):
            quant_config[f"model.layers.{i}.self_attn.q_proj"] = attn_quant_config
            quant_config[f"model.layers.{i}.self_attn.k_proj"] = attn_quant_config
            quant_config[f"model.layers.{i}.self_attn.v_proj"] = attn_quant_config
            quant_config[f"model.layers.{i}.self_attn.o_proj"] = attn_quant_config
            quant_config[f"model.layers.{i}.mlp.gate_proj"] = mlp_quant_config
            quant_config[f"model.layers.{i}.mlp.up_proj"] = mlp_quant_config
            quant_config[f"model.layers.{i}.mlp.down_proj"] = mlp_quant_config

        device_map: Dict[str, Any] = {}
        for name, _ in target_model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])
            if layer_name in quant_config:
                device_map[layer_name] = "cpu"
            else:
                device_map[layer_name] = device
        for name, _ in target_model.named_buffers():
            layer_name = ".".join(name.split(".")[:-1])
            device_map[layer_name] = device

        if "lm_head" not in device_map:
            device_map["lm_head"] = device

        target_config = {"device_map": device_map}
        draft_config = {
            "quant_config": {
                "processor": self.processor,
                "skip_modules": self.skip_modules,
            }
        }
        return target_config, draft_config
