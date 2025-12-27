#!/usr/bin/env bash
#
# Usage:
#   ./run.sh <python script>
#
# Example:
#   ./run.sh eagle.py
#
# Description:
#   This script adds numerous flags before running the selected Python script.

# Optional: activate the requested conda env (works in non-interactive shells).
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
  conda activate ~/envs/subspec >/dev/null 2>&1 || true
fi

CUDA_VISIBLE_DEVICES="0,1,2,3"
LOGLEVEL=INFO

# NVTX_PROFILING=True
NVTX_PROFILING=False

DETAILED_ANALYSIS=False

###############################################################################
# Construct command
###############################################################################
CMD="LOGLEVEL=$LOGLEVEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES DETAILED_ANALYSIS=$DETAILED_ANALYSIS PYTORCH_ALLOC_CONF=expandable_segments:True"

# Note: CUBLAS_WORKSPACE_CONFIG is configured in run/main.py.

if [ "$NVTX_PROFILING" = True ]; then
  # https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
  CMD+=" nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace=all --force-overwrite=true --python-sampling-frequency=1000 --python-sampling=true --cuda-memory-usage=true --gpuctxsw=true --python-backtrace -x true -o nsight_report"
fi

CMD+=" python -m $@"

###############################################################################
# Execute command
###############################################################################
echo "$CMD"
eval "$CMD"