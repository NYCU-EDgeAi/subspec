# SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLMs

The following code is modified from the original code in the [Sinq](https://github.com/huawei-csl/SINQ) library. See their [paper](https://www.arxiv.org/abs/2509.22944) for more information.

----

SINQ (Sinkhorn-Normalized Quantization) is a novel, fast and high-quality quantization method designed to make any Large Language Models smaller while keeping their accuracy almost intact.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/huawei-csl/SINQ.git
cd SINQ

# 2. Install dependencies
# pip install -r req.txt
pip install gemlite==0.5.1.post1

# 3. Install SINQ
pip install .
```