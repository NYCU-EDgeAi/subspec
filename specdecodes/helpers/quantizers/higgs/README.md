# HIGGS

The following code is modified from the HIGGS implementation in transformers. See the [transformers HIGGS docs](https://github.com/BlackSamorez/transformers/blob/53e6827faf3a2b660b4a330c5902e22ec75feef9/docs/source/en/quantization/higgs.md) for more information.

----

HIGGS is a 0-shot quantization algorithm that combines Hadamard preprocessing with MSE-Optimal quantization grids to achieve lower quantization error and SOTA performance. You can find more information in the paper [arxiv.org/abs/2411.17525](https://arxiv.org/abs/2411.17525).

Runtime support for HIGGS is implemented through [FLUTE](https://arxiv.org/abs/2407.10960), and its [library](https://github.com/HanGuo97/flute).

### Installation
```bash
pip install flute-kernel

# Install the fast-hadamard-transform library
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e .
```