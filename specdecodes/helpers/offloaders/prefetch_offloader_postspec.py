import torch
from torch import nn
from ..model_layer_orders import MODEL_TYPE_GET_LAYER_ORDER
from ..utils import find_child, get_tensors, check_device_map

class ChunkedPinMemory:
    """
    Splits a tensor into power-of-2 pinned-memory chunks,
    ensuring chunk size doesn't fall below a specified minimum size (1MB default).
    """
    def __init__(self, org_tensor, min_chunk_bytes=1 * 1024 * 1024):
        self.shape, self.dtype = org_tensor.shape, org_tensor.dtype

        flat = org_tensor.view(-1).cpu().detach()
        numel, elem_size = flat.numel(), flat.element_size()

        self.chunks = []
        self.sizes = []

        offset = 0
        while numel > 0:
            # Choose largest power-of-2 chunk <= remaining elements
            chunk_size = 1 << ((numel - 1).bit_length() - 1)
            # Enforce minimum chunk size
            if chunk_size * elem_size < min_chunk_bytes:
                chunk_size = numel

            chunk = torch.empty(chunk_size, dtype=self.dtype, pin_memory=True)
            chunk.copy_(flat[offset : offset + chunk_size])

            self.chunks.append(chunk)
            self.sizes.append(chunk_size)

            offset += chunk_size
            numel -= chunk_size

    def copy_to(self, gpu_tensor, non_blocking=False):
        """
        Copy chunked data back into a matching GPU tensor.
        """
        if gpu_tensor.numel() != sum(self.sizes):
            raise ValueError("GPU tensor size mismatch.")

        flat_gpu = gpu_tensor.view(-1)
        offset = 0
        for chunk, size in zip(self.chunks, self.sizes):
            flat_gpu[offset : offset + size].copy_(chunk, non_blocking=non_blocking)
            offset += size

def trim_layer_number(name: str) -> str:
    """Removes numeric fields from a dotted path (e.g. 'layer.0' -> 'layer')."""
    return ".".join(x for x in name.split(".") if not x.isdigit())

class PrefetchOffloader:
    def __init__(self, model: nn.Module, device_map: dict, draft_model: nn.Module = None):
        check_device_map(model, device_map)
        self.gpu_device = device_map["model.embed_tokens"]
        self.cpu_tensors = {}
        self.stream = torch.cuda.Stream()
        self.draft_model = draft_model

        self.draft_stream = torch.cuda.Stream()
        self.draft_done_event = torch.cuda.Event(blocking=False, interprocess=False)

        # Per-module copy completion events. Keyed by module object.
        self._copy_event_by_module = {}
        # Avoid launching draft work multiple times for the same pending copy event.
        self._draft_launched_for_copy_event = set()
        self._draft_in_flight = False

        self._cache_cpu_layers(model, device_map)
        assert model.model.embed_tokens.weight.device.type == "cuda"

        # V3 Prefetch Strategy: (V4: inline copy version)
        # 1. Load the first CPU layer to GPU, This layer will prefetch the next CPU layer
        # 2. The next CPU layer will prefetch the next CPU layer, and so on
        # 3. The last CPU layer will prefetch the first CPU layer

        layer_order = MODEL_TYPE_GET_LAYER_ORDER[model.config.model_type](model.config)
        cpu_layer_order = [name for name in layer_order if device_map.get(name) == "cpu"]

        # Find first 'cpu' layer
        if cpu_layer_order == []:
            raise ValueError("No CPU layer found in the model.")
        first_name = cpu_layer_order[0]
        first_cpu_layer = find_child(model, first_name)

        # Copy the first CPU layer to GPU
        for p, c in zip(get_tensors(first_cpu_layer), self.cpu_tensors[first_name]):
            # p.data.copy_(c)
            c.copy_to(p.data)

        i = 1
        current_layer = first_cpu_layer
        while i < len(cpu_layer_order):
            name = cpu_layer_order[i]
            next_layer = find_child(model, name)
            
            # Bundle the full MLP block (gate/up/down) together.
            # Trigger on gate_proj so the preceding module is typically self_attn.o_proj.
            if name.endswith(".mlp.gate_proj"):
                prefix = name.split(".mlp.")[0] + ".mlp"
                gate_name = prefix + ".gate_proj"
                up_name = prefix + ".up_proj"
                down_name = prefix + ".down_proj"

                gate_mod = find_child(model, gate_name)
                up_mod = find_child(model, up_name)
                down_mod = find_child(model, down_name)

                mlp_modules = [
                    (gate_mod, self.cpu_tensors[gate_name]),
                    (up_mod, self.cpu_tensors[up_name]),
                    (down_mod, self.cpu_tensors[down_name]),
                ]

                current_layer.register_forward_pre_hook(self._create_wait_hook())
                current_layer.register_forward_pre_hook(self._create_mlp_kickoff_hook(mlp_modules))

                # Wait hooks for the actual MLP modules. Draft work is launched (once) when gate_proj
                # is about to run and its copy is still pending.
                gate_mod.register_forward_pre_hook(self._create_wait_hook(maybe_run_draft=True))
                up_mod.register_forward_pre_hook(self._create_wait_hook())
                down_mod.register_forward_pre_hook(self._create_wait_hook())

                current_layer = down_mod
                i += 3
                continue

            # non-MLP layer processing
            current_layer.register_forward_pre_hook(self._create_wait_hook())
            current_layer.register_forward_pre_hook(self._create_prefetch_hook(next_layer, self.cpu_tensors[name]))
            current_layer = next_layer
            i += 1
            
            # print("prefetch non-MLP:", name)
            

        current_layer.register_forward_pre_hook(self._create_wait_hook())
        current_layer.register_forward_pre_hook(
            self._create_prefetch_hook(first_cpu_layer, self.cpu_tensors[first_name])
        )

        # Ensure any in-flight draft work is ordered before the caller proceeds.
        model.register_forward_hook(self._create_model_end_hook())
        
        # print("prefetch loop back to first layer:", first_name)
        
        
    
    def _cache_cpu_layers(self, model, device_map):
        """Moves CPU layers to pinned memory and creates GPU-shaped placeholders."""
        tensor_cache = {}
        for name, dev_str in device_map.items():
            layer = find_child(model, name)
            if dev_str == "cpu":
                trimmed = trim_layer_number(name)
                if trimmed not in tensor_cache:
                    placeholders = [torch.zeros_like(p, device=self.gpu_device, memory_format=torch.contiguous_format)
                                    for p in get_tensors(layer)]
                    tensor_cache[trimmed] = placeholders

                pinned = []
                for i, p in enumerate(get_tensors(layer)):
                    pinned.append(ChunkedPinMemory(p.data))
                    p.data = tensor_cache[trimmed][i]
                    
                self.cpu_tensors[name] = pinned
            else:
                # Move params/buffers directly to their specified device
                for p in get_tensors(layer):
                    p.data = p.data.to(dev_str)

    def _create_prefetch_hook(self, next_layer: nn.Module, cpu_params):
        """Schedules async CPU->GPU copy for `next_layer` immediately after current layer's forward."""
        def hook(module, inputs):
            with torch.cuda.stream(self.stream):
                for p, c in zip(get_tensors(next_layer), cpu_params):
                    # p.data.copy_(c)
                    c.copy_to(p.data, non_blocking=True)

                evt = torch.cuda.Event(blocking=False, interprocess=False)
                evt.record(self.stream)
                self._copy_event_by_module[next_layer] = evt
        
        return hook
    
    def _create_pre_draft_hook(self):
        def hook(module, inputs):
            self.draft_model.postspec()
        return hook
    
    def _create_wait_hook(self, maybe_run_draft: bool = False):
        """Wait for this module's weights via CUDA events (no full-device sync)."""
        def hook(module, inputs):
            evt = self._copy_event_by_module.get(module)
            if evt is None:
                return

            if maybe_run_draft and self.draft_model is not None and not evt.query():
                evt_id = id(evt)
                if evt_id not in self._draft_launched_for_copy_event:
                    self._draft_launched_for_copy_event.add(evt_id)
                    self._launch_draft_postspec_while_copy_pending(evt)

            torch.cuda.current_stream().wait_event(evt)

        return hook
    
    def _create_mlp_kickoff_hook(self, mlp_modules):
        def hook(module, inputs):
            with torch.cuda.stream(self.stream):
                for submod, cpu_params in mlp_modules:
                    for p, c in zip(get_tensors(submod), cpu_params):
                        c.copy_to(p.data, non_blocking=True)

                evt = torch.cuda.Event(blocking=False, interprocess=False)
                evt.record(self.stream)
                for submod, _ in mlp_modules:
                    self._copy_event_by_module[submod] = evt
        return hook

    def _launch_draft_postspec_while_copy_pending(self, copy_evt: torch.cuda.Event, max_steps: int = 8):
        """Enqueue several postspec steps while the MLP copy is pending."""
        with torch.cuda.stream(self.draft_stream):
            steps = 0
            while steps < max_steps and not copy_evt.query():
                self.draft_model.postspec()
                steps += 1
            if steps > 0:
                self.draft_done_event.record(self.draft_stream)
                self._draft_in_flight = True

    def _create_model_end_hook(self):
        def hook(module, inputs, output):
            if self._draft_in_flight:
                torch.cuda.current_stream().wait_event(self.draft_done_event)

            self._draft_in_flight = False
            self._draft_launched_for_copy_event.clear()
            return output
        return hook