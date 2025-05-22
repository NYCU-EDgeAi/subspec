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

        layer_order = MODEL_TYPE_GET_LAYER_ORDER[model.config.model_type](model.config)
        
        # V3 Prefetch Strategy:
        # 1. Load the first CPU layer to GPU, This layer will prefetch the next CPU layer
        # 2. The next CPU layer will prefetch the next CPU layer, and so on
        # 3. The last CPU layer will prefetch the first CPU layer
        
        # Find first 'cpu' layer
        first_idx = next((i for i, name in enumerate(layer_order) if device_map.get(name) == "cpu"), -1)
        if first_idx == -1:
            raise ValueError("No CPU layer found in the model.")
        first_cpu_layer = find_child(model, layer_order[first_idx])
        
        # Load the first CPU layer to GPU
        for p in get_tensors(first_cpu_layer):
            p.data = p.data.to(self.gpu_device)
        
        # Connect subsequent CPU layers in a chain
        current_layer = first_cpu_layer
        for name in layer_order[first_idx+1:]:
            if device_map.get(name) == "cpu":
                next_layer = find_child(model, name)
                current_layer.register_forward_pre_hook(self._create_prefetch_hook(next_layer))
                current_layer = next_layer

                current_layer.register_forward_pre_hook(self._create_wait_hook())
                current_layer.register_forward_hook(
                    self._create_offload_hook(current_layer, self.cpu_tensors[name])
                )

        # Connect the last CPU layer to the first CPU layer
        if current_layer != first_cpu_layer: # If there is only one CPU layer, no need to prefetch
            # Set up pre-hook (wait for copy) and post-hook (offload) for the first CPU layer
            first_cpu_layer.register_forward_pre_hook(self._create_wait_hook(), prepend=True) # Prepend to ensure wait runs before prefetch
            first_cpu_layer.register_forward_hook(
                self._create_offload_hook(first_cpu_layer, self.cpu_tensors[layer_order[first_idx]])
            )
            # The last CPU layer prefetches the first CPU layer (forming a loop)
            current_layer.register_forward_pre_hook(self._create_prefetch_hook(first_cpu_layer))
                
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