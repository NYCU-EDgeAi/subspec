import logging
import importlib.util
from hqq.core.quantize import *
from hqq.utils.patching import prepare_for_inference
from .hf.base import AutoHQQHFModel

# Check for the kernel library (replace 'kernel_library' with the actual module name)
if importlib.util.find_spec("gemlite") is None:
    raise ImportError(
        "The 'gemlite' kernel is required for optimized performance. "
        "Please install it using 'pip install git+https://github.com/mobiusml/gemlite/'."
    )

class HqqQuantizer:
    @classmethod
    def quantize_model(cls, model, quant_config, compute_dtype, device):
        logging.info("Quantizing model with HqqQuantizer")
        AutoHQQHFModel.quantize_model(model, quant_config, compute_dtype=compute_dtype, device=device)
        HQQLinear.set_backend(HQQBackend.PYTORCH)
        prepare_for_inference(model, backend="gemlite")