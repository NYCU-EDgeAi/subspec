# Substitute Speculative Decoding (SubSpec)

This repository is the official implementation of *"Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding"*.

![fig1](./assets/fig1.png)

## Requirements

First, create and activate a conda environment with the following command:

```bash
conda create -n subspec python=3.11
conda activate subspec
```

Then, install [PyTorch](https://pytorch.org/get-started/locally/) from the official website. 

Install the rest of the base requirements:

```setup
pip install "smolagents[toolkit]"
pip install -r requirements.txt
```

You will need to install the additional libraries for quantization:

- HQQ (Default)
```bash
pip install hqq
pip install gemlite==0.5.1.post1
```
- HIGGS (optional)
```bash
pip install flute-kernel

# Install the fast-hadamard-transform library
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e .
```

## Evaluation

To evaluate the performance of SubSpec and other methods, use the unified entry point `run.main`:

```bash
# Basic usage
python -m run.main --method <method_name> run-test

# Running detailed benchmarks
python -m run.main --method <method_name> run-benchmark --benchmarks <benchmarks> --max-samples 20
```

### Available Methods
The following methods are available (registered in `run/core/registers.py`):
- `subspec_sd`: Substitute Speculative Decoding (Offloading + HQQ Quantization)
- `classic_sd`: Standard Speculative Decoding
- `vanilla`: Base LLM inference (no speculative decoding)
- `eagle_sd`: EAGLE Speculative Decoding
- ...and others (`subspec_sd_v2`, `subspec_sd_no_offload`, etc.)

### Common Arguments
- `--method`: The decoding method to use (required for defaults).
- `--device`: Target device (e.g., `cuda:0`, `cuda:1`). Defaults to `cuda:0`.
- `--warmup-iter`: Number of warmup iterations. Default varies by method (typically 1).
- `--compile-mode`: Torch compile mode (e.g., `reduce-overhead`, `max-autotune`, or `none`). Defaults to `none` (or method-specific default).

### Examples

**1. Evaluate SubSpec on MT-Bench with specific GPU:**
```bash
python -m run.main --method subspec_sd --device "cuda:0" run-benchmark --benchmarks mt-bench --max-samples 20
```

**2. Run a quick test with Classic SD on a different GPU:**
```bash
python -m run.main --method classic_sd --device "cuda:1" --warmup-iter 0 run-test
```

**9. Selectable Benchmarks:**
"mt-bench", "human-eval", "gsm8k", "alpaca", "cnn-dm", "aime", "gpqa", "math-500", and "livecodebench".

> The datasets and pretrained models will be downloaded automatically from Hugging Face.

## Results

SubSpec achieves superior performance on various benchmarks. 

Below is the result for accelerating Qwen2.5 7B with tree-based speculative decoding using different draft models, running 20 samples on MT-Bench:

| Draft Model        | tokens/sec | τ |
| ------------------ |---------------- | -------------- |
| [EAGLE-2](https://huggingface.co/leptonai/EAGLE-Qwen2.5-7B-Instruct)      |      7.56        |      3.90      |
| Qwen2.5 1.5B  |      15.14       |      11.91     |
| SubSpec       |    **24.29**     |   **28.35**    |

> τ represents average acceptance length, which is the the mean number of the accepted draft tokens per iteration.


> For EAGLE's draft model, you will need to download the pretrained model manually, then convert it with the 'convert_eagle_weights.ipynb' script before use.