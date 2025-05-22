# Modified from https://gist.github.com/gau-nernst/9408e13c32d3c6e7025d92cce6cba140
import torch
from torch import nn
from ..model_layer_orders import MODEL_TYPE_GET_LAYER_ORDER

def find_child(model, name: str) -> nn.Module:
    module_tree = name.split(".")
    parent = model
    for m in module_tree:
        # parent = parent._modules[m]
        parent = getattr(parent, m)
    return parent


def check_device_map(model: nn.Module, device_map: dict):
    """
    Checks a device map covers everything in a given model.

    Args:
        model (`torch.nn.Module`): The model to check the device map against.
        device_map (`Dict[str, Union[int, str, torch.device]]`): The device map to check.
    """
    all_model_tensors = [name for name, _ in model.state_dict().items()]
    for module_name in device_map.keys():
        if module_name == "":
            all_model_tensors.clear()
            break
        else:
            all_model_tensors = [
                name
                for name in all_model_tensors
                if not name == module_name and not name.startswith(module_name + ".")
            ]
    if len(all_model_tensors) > 0:
        non_covered_params = ", ".join(all_model_tensors)
        raise ValueError(
            f"The device_map provided does not give any device for the following parameters: {non_covered_params}"
        )

class Offloader:
    def __init__(self, model, device_map, record_stream=False):  
        # Debugging, check if device_map covers all tensors in the model
        check_device_map(model, device_map)
        
        # Pin the CPU parameters/buffers to the memory, and create GPU tensors for the specified parameters/buffers on device_map.
        for name, device in device_map.items():
            layer = find_child(model, name)
            for param in layer.parameters():
                if device == "cpu":
                    param.data = param.data.cpu().pin_memory()
                else:
                    param.data = param.data.to(device)
                    
            for buffer in layer.buffers():
                if device == "cpu":
                    buffer.data = buffer.data.cpu().pin_memory()
                else:
                    buffer.data = buffer.data.to(device)
        
        # Ensure the first layer is already on GPU
        assert model.model.embed_tokens.weight.device.type == "cuda"
        gpu_device = model.model.embed_tokens.weight.device
        
        # Save the original parameters
        self.param_dict = {p: p.data for p in model.parameters()}
        
        # Create hooks on model, to pre-load the tensors to the GPU before the preceeding layer is executed.
        def create_pre_hook(cur_layer, device):
            @torch.compiler.disable()
            def pre_hook(module, args):
                for p in cur_layer.parameters():
                    p.data = p.data.to(device=device, non_blocking=False)
                        
            return pre_hook

        @torch.compiler.disable()
        def post_hook(module, args, output):
            torch.cuda.current_stream().synchronize()
            for p in module.parameters():
                p.data = self.param_dict[p]        
        
        # Replace traverse
        layer_order = MODEL_TYPE_GET_LAYER_ORDER[model.config.model_type](model.config)
        for i in range(len(layer_order)):
            cur_device = device_map.get(layer_order[i])
            current_layer = find_child(model, layer_order[i])
            
            if cur_device == "cpu":
                current_layer.register_forward_pre_hook(create_pre_hook(current_layer, device=gpu_device))
                current_layer.register_forward_hook(post_hook)