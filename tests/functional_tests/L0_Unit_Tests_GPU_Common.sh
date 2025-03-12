NEMO_NUMBA_MINVER=0.53 CUDA_VISIBLE_DEVICES=0 pytest tests/collections/common -m "not pleasefixme" --with_downloads --cov-branch --cov-report=xml --cov=nemo
