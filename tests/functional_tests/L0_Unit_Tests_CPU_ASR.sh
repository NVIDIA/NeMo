CUDA_VISIBLE_DEVICES="" NEMO_NUMBA_MINVER=0.53 pytest tests/collections/asr -m "not pleasefixme" --cpu --with_downloads --relax_numba_compat --cov-branch --cov-report=xml --cov=nemo
