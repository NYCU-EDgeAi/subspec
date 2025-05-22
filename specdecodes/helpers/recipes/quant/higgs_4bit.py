from ..base_recipe import QuantOffloadRecipe
from ...quantizers.higgs import HiggsQuantizer

class Recipe(QuantOffloadRecipe):
    def __init__(self):
        super().__init__()
        # Assign quantizer and offloader objects.
        self.quantizer = HiggsQuantizer
        self.offloader = None

    def generate_configurations(self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device):
        # Quantization
        quant_config = {}
        attn_quant_config = {
            "bits": 4,
            "p": 2,
            "group_size": 128,
            "hadamard_size": 512,
        }
        mlp_quant_config = {
            "bits": 4,
            "p": 2,
            "group_size": 128,
            "hadamard_size": 512,
        }
        
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

        # Configs
        target_config = {
            "quant_config": quant_config,
        }
        draft_config = None
        
        return target_config, draft_config