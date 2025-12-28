from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..base_recipe import QuantOffloadRecipe
from specdecodes.helpers.offloaders.offloader import Offloader


class LayerOffloadRecipe(QuantOffloadRecipe):
    """Offload all decoder layers after the first N to CPU.

    This matches the exp_offloading/* recipes that were duplicated per model, where the
    only varying knob was "how many layers are kept on GPU".
    """

    def __init__(self, keep_first_n_layers_on_gpu: int = 0):
        super().__init__()
        self.quantizer = None
        self.offloader = Offloader
        self.keep_first_n_layers_on_gpu = int(keep_first_n_layers_on_gpu)

    def generate_configurations(
        self,
        target_model: Any,
        draft_model: Any,
        max_length: int,
        cpu_offload_gb: Optional[int],
        dtype: Any,
        device: str,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        layer_cnt = len(target_model.model.layers)

        start = max(0, min(self.keep_first_n_layers_on_gpu, layer_cnt))
        device_config = {}
        for i in range(start, layer_cnt):
            device_config[f"model.layers.{i}.self_attn.q_proj"] = "cpu"
            device_config[f"model.layers.{i}.self_attn.k_proj"] = "cpu"
            device_config[f"model.layers.{i}.self_attn.v_proj"] = "cpu"
            device_config[f"model.layers.{i}.self_attn.o_proj"] = "cpu"
            device_config[f"model.layers.{i}.mlp.gate_proj"] = "cpu"
            device_config[f"model.layers.{i}.mlp.up_proj"] = "cpu"
            device_config[f"model.layers.{i}.mlp.down_proj"] = "cpu"

        device_map: Dict[str, str] = {}
        for name, _ in target_model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])
            device_map[layer_name] = "cpu" if layer_name in device_config else device
        for name, _ in target_model.named_buffers():
            layer_name = ".".join(name.split(".")[:-1])
            device_map[layer_name] = device

        target_config = {
            "device_map": device_map,
            "quant_config": None,
        }
        return target_config, None


# Convenience alias for YAML configs that prefer the conventional name.
Recipe = LayerOffloadRecipe
