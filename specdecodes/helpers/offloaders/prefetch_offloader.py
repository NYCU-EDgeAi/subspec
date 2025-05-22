import torch
from torch import nn
from ..model_layer_orders import MODEL_TYPE_GET_LAYER_ORDER
from ..utils import find_child, get_tensors, check_device_map


class PrefetchOffloader:
    def __init__(self, model, device_map, record_stream=False):
        check_device_map(model, device_map)
        self.gpu_device = device_map["model.embed_tokens"]
        self.cpu_tensors = {}
        self.stream = torch.cuda.Stream()
        self.record_stream = record_stream

        self._cache_cpu_layers(model, device_map)
        assert model.model.embed_tokens.weight.device.type == 'cuda'

        # Walk through layers in order, registering:
        #    (a) If the layer itself is CPU-based, attach:
        #       - A forward_pre_hook that waits for the copy to finish.
        #       - A forward_hook that offloads the layer back to CPU after forward.
        #    (b) If next layer is CPU-based, attach a forward_hook that enqueues copy -> self.gpu_device.
        layer_order = MODEL_TYPE_GET_LAYER_ORDER[model.config.model_type](model.config)
        for i, layer_name in enumerate(layer_order):
            current_layer = find_child(model, layer_name)
            current_dev_str = device_map.get(layer_name, 'cpu')

            # (a) If this layer is CPU-based, wait for the copy + offload back afterwards
            if current_dev_str == 'cpu':
                # Wait for the async copy right before forward
                current_layer.register_forward_pre_hook(self._create_wait_hook())
                # Move weights back to pinned CPU after forward
                current_layer.register_forward_hook(self._create_offload_hook(current_layer, self.cpu_tensors[layer_name]))

            # (b) Prefetch next CPU layer at the end of this layer's forward pass
            if i + 1 < len(layer_order):
                next_name = layer_order[i + 1]
                next_dev_str = device_map.get(next_name, 'cpu')
                if next_dev_str == 'cpu':
                    next_layer = find_child(model, next_name)
                    current_layer.register_forward_pre_hook(
                        self._create_prefetch_hook(next_layer)
                    )
                
    def _cache_cpu_layers(self, model, device_map):
        """Moves CPU layers to pinned memory and creates GPU-shaped placeholders."""
        for name, dev_str in device_map.items():
            layer = find_child(model, name)
            if dev_str == 'cpu':
                pinned = []
                for p in get_tensors(layer):
                    pinned.append(p.data.cpu().pin_memory())
                    p.data = pinned[-1]
                self.cpu_tensors[name] = pinned
            else:
                for p in get_tensors(layer):
                    p.data = p.data.to(dev_str)

    def _create_prefetch_hook(self, next_layer):
        """Schedules async CPU->GPU copy for `next_layer` immediately after current layer's forward."""
        @torch.compiler.disable()
        def hook(module, inputs):
            with torch.cuda.stream(self.stream):
                for p in get_tensors(next_layer):
                    p.data = p.data.to(self.gpu_device, non_blocking=True)
                    if self.record_stream:
                        p.data.record_stream(self.stream)
        return hook

    def _create_wait_hook(self):
        """Waits for any pending async copies in self.stream to finish before forward execution."""
        @torch.compiler.disable()
        def hook(module, inputs):
            torch.cuda.current_stream().wait_stream(self.stream)
        return hook

    def _create_offload_hook(self, cur_layer: nn.Module, cpu_params):
        """Restore pinned CPU memory parameters/buffers afterforward execution."""
        @torch.compiler.disable()
        def offload_hook(module, inputs, output):
            # If we're *not* using record_stream, we need to ensure the GPU is done
            # with the forward pass before modifying p.data
            if not self.record_stream:
                torch.cuda.current_stream().synchronize()
                
            for i, p in enumerate(get_tensors(cur_layer)):
                p.data = cpu_params[i]
            
            
        return offload_hook