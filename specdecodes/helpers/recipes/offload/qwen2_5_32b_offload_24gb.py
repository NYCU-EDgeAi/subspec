from ..base_recipe import QuantOffloadRecipe
from specdecodes.helpers.offloaders.offloader import Offloader

class Recipe(QuantOffloadRecipe):
    def __init__(self):
        super().__init__()
        # Assign quantizer and offloader objects.
        self.quantizer = None
        self.offloader = Offloader

    def generate_configurations(self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device):
        layer_cnt = len(target_model.model.layers)
    
        # Offloading
        device_config = {}
        start = 20 # First 20 layers are kept on GPU
        end = layer_cnt
        for i in range(start, end):
            device_config[f"model.layers.{i}.self_attn.q_proj"] = 'cpu'
            device_config[f"model.layers.{i}.self_attn.k_proj"] = 'cpu'
            device_config[f"model.layers.{i}.self_attn.v_proj"] = 'cpu'
            device_config[f"model.layers.{i}.self_attn.o_proj"] = 'cpu'
            device_config[f"model.layers.{i}.mlp.gate_proj"] = 'cpu'
            device_config[f"model.layers.{i}.mlp.up_proj"] = 'cpu'
            device_config[f"model.layers.{i}.mlp.down_proj"] = 'cpu'
    
        # Set device map
        device_map = {}
        for name, _ in target_model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])
            if layer_name in device_config:
                device_map[layer_name] = 'cpu'
            else:
                device_map[layer_name] = device
        for name, _ in target_model.named_buffers():
            layer_name = ".".join(name.split(".")[:-1])
            device_map[layer_name] = device

        # Configs
        target_config = {
            "device_map": device_map,
            "quant_config": None,
        }
        draft_config = None
        
        return target_config, draft_config