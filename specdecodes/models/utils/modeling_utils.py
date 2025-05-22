import torch.nn as nn

def get_named_tensors(
    module: nn.Module, 
    recurse: bool = False
):
    yield from module.named_parameters(recurse=recurse)
    yield from module.named_buffers(recurse=recurse)