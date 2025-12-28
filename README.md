# Substitute Speculative Decoding (SubSpec)

This repository is the official implementation of *"Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding"*.

![fig1](./assets/fig1.png)

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
	- [YAML configs (recommended)](#yaml-configs-recommended)
	- [Available Methods](#available-methods)
	- [Common Arguments](#common-arguments)
- [Evaluation](#evaluation)
	- [Examples](#examples)
- [Results](#results)
- [Interfaces](#interfaces)
	- [Gradio Demo](#gradio-demo)
	- [OpenAI-Compatible API Server](#openai-compatible-api-server)
		- [Run the server](#run-the-server)
		- [Example: eagle_sd](#example-eagle_sd)
		- [Quick API checks](#quick-api-checks)
- [Testing](#testing)

## Requirements

First, create and activate a conda environment with the following command:

```bash
conda create -n subspec python=3.11
conda activate subspec
```

Then, install [PyTorch](https://pytorch.org/get-started/locally/) from the official website. 

Install the rest of the base requirements:

```bash
pip install "smolagents[toolkit]"
pip install -r requirements.txt
```

If you want to use the Gradio UI:

```bash
pip install gradio
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
cd ..
```

## Usage

All entrypoints go through `run.main` by setting a YAML config (`--config`).

### YAML configs (recommended)

- Method templates: `configs/methods/`
- Migrated offloading experiment configs: `configs/exp_offloading/`

Precedence: method defaults < YAML < CLI.

```bash
# Run from a method YAML
python -m run.main --config configs/methods/classic_sd.yaml run-test

# Run from an exp_offloading YAML (includes a recipe + per-model offload settings)
python -m run.main --config configs/exp_offloading/vanilla_qwen_7b.yaml run-test

# Override YAML values from the CLI
python -m run.main --config configs/methods/classic_sd.yaml --device cuda:1 --warmup-iter 0 run-test
```

Offloading YAML configs parameterize “how many layers remain on GPU” via:

```yaml
recipe:
  class_path: specdecodes.helpers.recipes.offload.layer_offload:LayerOffloadRecipe
  kwargs:
    keep_first_n_layers_on_gpu: <N>
```

### Available Methods
The following methods are available (registered in `run/core/presets.py`):
- `subspec_sd`: Substitute Speculative Decoding (Offloading + HQQ Quantization)
- `classic_sd`: Standard Speculative Decoding
- `vanilla`: Base LLM inference (no speculative decoding)
- `eagle_sd`: EAGLE Speculative Decoding
- ...and others (`subspec_sd_v2`, `subspec_sd_no_offload`, etc.)

### Common Arguments
- `--config`: Path to a YAML config (required). CLI args override YAML.
- `--method`: Optional override for the YAML `method`.
- `--device`: Target device (e.g., `cuda:0`, `cuda:1`). Defaults to `cuda:0`.
- `--warmup-iter`: Number of warmup iterations. Default varies by method (typically 1).
- `--compile-mode`: Torch compile mode (e.g., `reduce-overhead`, `max-autotune`, or `none`). Defaults to `none`.

## Evaluation

```bash
# Quick sanity check
python -m run.main --config configs/methods/<method_name>.yaml run-test

# Detailed benchmark run
python -m run.main --config configs/methods/<method_name>.yaml run-benchmark --benchmarks <benchmarks> --max-samples 20
```

### Examples

**1. Evaluate SubSpec on MT-Bench with specific GPU:**
```bash
python -m run.main --config configs/methods/subspec_sd.yaml --device "cuda:0" run-benchmark --benchmarks mt-bench --max-samples 20
```

**2. Run a quick test with Classic SD on a different GPU:**
```bash
python -m run.main --config configs/methods/classic_sd.yaml --device "cuda:1" --warmup-iter 0 run-test
```

**Selectable benchmarks:**
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

## Interfaces

### Gradio Demo

Launch an interactive chat UI:

```bash
python -m run.main --config configs/methods/<method_name>.yaml run-gradio --host 127.0.0.1 --port 7860
```

To expose it on your network (or use a public share link):

```bash
python -m run.main --config configs/methods/<method_name>.yaml run-gradio --host 0.0.0.0 --port 7860
python -m run.main --config configs/methods/<method_name>.yaml run-gradio --share
```

### OpenAI-Compatible API Server

This repo includes an OpenAI-compatible HTTP server implemented in [run/pipelines/run_api.py](run/pipelines/run_api.py).

#### Run the server

1) Activate your environment:

```bash
conda activate ~/envs/subspec
```

2) Start the server via the unified entry point:

```bash
python -m run.main --config configs/methods/<method_name>.yaml run-api --host 0.0.0.0 --port 8000
```

#### Example: `eagle_sd`

`eagle_sd` requires both a target model (`--llm-path`) and a draft model (`--draft-model-path`). The default draft path in the preset is:

```text
~/checkpoints/eagle/official/EAGLE-Llama-3.1-8B-Instruct
```

Run:

```bash
python -m run.main \
	--config configs/methods/eagle_sd.yaml \
	--llm-path meta-llama/Llama-3.1-8B-Instruct \
	--draft-model-path ~/checkpoints/eagle/official/EAGLE-Llama-3.1-8B-Instruct \
	--device cuda:0 \
	--warmup-iter 0 \
	run-api --host 0.0.0.0 --port 8000
```

#### Quick API checks

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

Chat completions (non-stream):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{"messages":[{"role":"user","content":"Say OK"}],"max_tokens":16,"temperature":0}' | jq .
```

Chat completions (stream):

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{"messages":[{"role":"user","content":"Count to three"}],"max_tokens":32,"temperature":0,"stream":true}'
```

## Testing

Run the unit tests with:

```bash
pytest -q
```

Some tests that run real model workloads are skipped by default. To enable them:

```bash
SUBSPEC_RUN_REAL_MODEL_TESTS=1 pytest -q
```