from ..base_recipe import QuantOffloadRecipe
from hqq.core.quantize import *
from ...quantizers.hqq import HqqQuantizer
from ...offloaders.prefetch_offloader_v5 import PrefetchOffloader


class Recipe(QuantOffloadRecipe):
    def __init__(self):
        super().__init__()
        # Assign quantizer and offloader objects.
        self.quantizer = HqqQuantizer
        self.offloader = PrefetchOffloader

    def generate_configurations(self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device):
        # Quantization
        quant_config = {}
        attn_quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
        mlp_quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
        
        layer_cnt = len(target_model.model.layers)
        quant_start = 0
        quant_end = layer_cnt
        for i in range(quant_start, quant_end):
            quant_config[f"model.layers.{i}.self_attn.q_proj"] = attn_quant_config
            quant_config[f"model.layers.{i}.self_attn.k_proj"] = attn_quant_config
            quant_config[f"model.layers.{i}.self_attn.v_proj"] = attn_quant_config
            quant_config[f"model.layers.{i}.self_attn.o_proj"] = attn_quant_config
            quant_config[f"model.layers.{i}.mlp.gate_proj"] = mlp_quant_config
            quant_config[f"model.layers.{i}.mlp.up_proj"] = mlp_quant_config
            quant_config[f"model.layers.{i}.mlp.down_proj"] = mlp_quant_config
            
        # Device map
        device_map = {}
        for name, _ in target_model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])
            if layer_name in quant_config:
                device_map[layer_name] = 'cpu'
            else:
                device_map[layer_name] = device
        for name, _ in target_model.named_buffers():
            layer_name = ".".join(name.split(".")[:-1])
            device_map[layer_name] = device
        
        # Hotfix for Llama-3.1-8B-Instruct
        if "lm_head" not in device_map:
            device_map["lm_head"] = device

        # Configs
        target_config = {
            "device_map": device_map,
        }
        draft_config = {
            "quant_config": quant_config,
        }
        
        return target_config, draft_config