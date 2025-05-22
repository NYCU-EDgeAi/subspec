import torch
from torch import nn
from ..model_layer_orders import MODEL_TYPE_GET_LAYER_ORDER
from ..utils import find_child, get_tensors, check_device_map


def trim_layer_number(name: str) -> str:
    """Removes numeric fields from a dotted path (e.g. 'layer.0' -> 'layer')."""
    return ".".join(x for x in name.split(".") if not x.isdigit())

class PrefetchOffloader:
    def __init__(self, model: nn.Module, device_map: dict, record_stream=False):
        check_device_map(model, device_map)
        self.gpu_device = device_map["model.embed_tokens"]
        self.cpu_tensors = {}
        self.stream = torch.cuda.Stream()
        self.record_stream = record_stream

        self._cache_cpu_layers(model, device_map)
        assert model.model.embed_tokens.weight.device.type == "cuda"

        # Walk through layers in order, registering:
        #    (a) If the layer itself is CPU-based, attach:
        #       - A forward_pre_hook that waits for the copy to finish.
        #    (b) If next layer is CPU-based, attach a forward_hook that enqueues copy -> self.gpu_device.
        layer_order = MODEL_TYPE_GET_LAYER_ORDER[model.config.model_type](model.config)
        for i, layer_name in enumerate(layer_order):
            current_layer = find_child(model, layer_name)
            current_dev_str = device_map.get(layer_name, "cpu")
            
            # (a) If current layer is CPU-based, wait for async copy before forward
            if current_dev_str == "cpu":
                # Wait for the async copy right before forward
                current_layer.register_forward_pre_hook(self._create_wait_hook())

            # (b) Hook to prefetch next CPU-based layer at the end of current layer's forward pass
            if i + 1 < len(layer_order):
                next_name = layer_order[i + 1]
                if device_map.get(next_name, "cpu") == "cpu":
                    next_layer = find_child(model, next_name)
                    current_layer.register_forward_hook(
                        self._create_prefetch_hook(next_layer, self.cpu_tensors[next_name])
                    )


    def _cache_cpu_layers(self, model, device_map):
        """Moves CPU layers to pinned memory and creates GPU-shaped placeholders."""
        tensor_cache = {}
        for name, dev_str in device_map.items():
            layer = find_child(model, name)
            if dev_str == "cpu":
                trimmed = trim_layer_number(name)
                if trimmed not in tensor_cache:
                    placeholders = [torch.zeros_like(p, device=self.gpu_device)
                                    for p in get_tensors(layer)]
                    tensor_cache[trimmed] = placeholders

                pinned = []
                for i, p in enumerate(get_tensors(layer)):
                    pinned.append(p.data.cpu().pin_memory())
                    p.data = tensor_cache[trimmed][i]
                self.cpu_tensors[name] = pinned
            else:
                # Move params/buffers directly to their specified device
                for p in get_tensors(layer):
                    p.data = p.data.to(dev_str)

    def _create_prefetch_hook(self, next_layer: nn.Module, cpu_params):
        """Schedules async CPU->GPU copy for `next_layer` immediately after current layer's forward."""
        def hook(module, inputs, output):
            with torch.cuda.stream(self.stream):
                for p, c in zip(get_tensors(next_layer), cpu_params):
                    p.data.copy_(c, non_blocking=True)
                    if self.record_stream:
                        p.data.record_stream(torch.cuda.current_stream())
        return hook

    def _create_wait_hook(self):
        """Waits for any pending async copies in self.stream to finish before forward execution."""
        def hook(module, inputs):
            torch.cuda.current_stream().wait_stream(self.stream)
        return hook