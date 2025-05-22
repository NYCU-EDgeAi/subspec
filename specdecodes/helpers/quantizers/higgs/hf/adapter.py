import copy

import torch
import torch.nn as nn
from torch import float16
from typing import Callable, Optional, Union
from termcolor import colored
from transformers.integrations import HiggsLinear, quantize_with_higgs

def quantize_param(
    module_name,
    module,
    param_value,
    quant_config,
    tune_metadata,
    target_device,
):
    """
    Quantizes weights into weight and weight_scale
    """
    flute_dict = quantize_with_higgs(
        param_value.to(target_device),
        quant_config["bits"],
        quant_config['p'],
        quant_config["group_size"],
        quant_config["hadamard_size"],
    )
    del param_value

    for key, value in flute_dict.items():
        if key in module._parameters:
            # module._parameters[key] = torch.nn.Parameter(value, requires_grad=False)
            #update the parameter value
            module._parameters[key].data = value
        elif key in module._buffers:
            # module._buffers[key] = torch.nn.Buffer(value)
            module._buffers[key].data = value
        elif key == "tune_metadata":
            module.tune_metadata = value
            tune_metadata[module_name] = value.to_dict()
        else:
            raise ValueError(f"Unexpected key {key} in module {module}")
        

def higgs_linear_adapter(linear_layer, quant_config, tune_metadata, device, compute_dtype):
    if linear_layer is not None:
        module_name = linear_layer.name
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # Quantize the weights
        out_module = HiggsLinear(
            in_features,
            out_features,
            bias=linear_layer.bias is not None,
            num_bits=quant_config["bits"],
            group_size=quant_config["group_size"],
            hadamard_size=quant_config["hadamard_size"],
            dtype=compute_dtype,
            device=device,
        )
        quantize_param(module_name, out_module, linear_layer.weight.data, quant_config, tune_metadata, device)
        return out_module
    
    return None