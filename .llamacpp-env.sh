# CUDA toolkit assembled from pip wheels in .buildcuda (no sudo, no system CUDA).
# Kept separate from .venv so vLLM's pinned CUDA runtime is never disturbed.
export CT=/home/wylin/ai-academic-advisor/.cudatoolkit
export CUDACXX=$CT/bin/nvcc
export CUDAToolkit_ROOT=$CT
export PATH=$CT/bin:$PATH
export LD_LIBRARY_PATH=$CT/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CT/lib64:$LIBRARY_PATH
