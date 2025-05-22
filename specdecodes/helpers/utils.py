import torch
import torch.nn as nn


def find_child(model: nn.Module, name: str) -> nn.Module:
    """Recursively access submodules by dotted path (e.g. 'model.layer.0')."""
    for part in name.split("."):
        model = getattr(model, part)
    return model

def get_tensors(module: nn.Module):
    """Yields module parameters and buffers."""
    yield from module.parameters()
    yield from module.buffers()
    
def get_named_tensors(module: nn.Module):
    """Yields module named parameters and buffers."""
    yield from module.named_parameters()
    yield from module.named_buffers()
        
def estimate_quantized_size(model, quant_config, max_input_len=0):
    weight_bytes = 0
    for name, param in get_named_tensors(model):
        layer_name = ".".join(name.split(".")[:-1])
        if layer_name in quant_config:
            nbits = quant_config[layer_name]['weight_quant_params']['nbits']
            group_size = quant_config[layer_name]['weight_quant_params']['group_size']
            weight_bytes += param.numel() * param.element_size() * nbits / 16 # quantized weight
            weight_bytes += param.numel() * param.element_size() / group_size * 2 # scale and zero
        else:
            weight_bytes += param.numel() * param.element_size()
    
    # key and value cache
    kv_bytes = 0
    if max_input_len > 0:
        element_size = next(iter(model.parameters())).element_size() # assume activation has same element size as first param
        head_size = model.config.hidden_size // model.config.num_attention_heads
        kv_bytes = 2 * max_input_len * model.config.num_hidden_layers * model.config.num_key_value_heads * head_size * element_size # key and value cache
    
    return weight_bytes + kv_bytes

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