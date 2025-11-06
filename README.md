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

To evaluate the performance of SubSpec, you can use the provided `run.sh` script.
```eval
bash run.sh <pipeline path> run-benchmark --benchmarks <benchmarks> --max-samples 20
```
All experiment pipelines conducted in this work are located in the run/exp_offload directory.

For example, to evaluate SubSpec on Qwen2.5 7B Instruct on MT-Bench and GSM8K, run:
```bash
bash run.sh run.exp_offloading.subspec_sd_qwen_7b run-benchmark --benchmarks mt-bench,gsm8k --max-samples 20
```

There are a total of 9 selectable benchmarks:
"mt-bench", "human-eval", "gsm8k", "alpaca", "cnn-dm", "aime", "gpqa", "math-500", and "livecodebench".

> the datasets and pretrained models will be downloaded automatically from Hugging Face.

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