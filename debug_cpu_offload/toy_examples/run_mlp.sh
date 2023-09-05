echo "no offloading..."
python mlp.py

echo "jit-offloading..."
python mlp_offload_jit.py

echo "prefetch-offloading..."
python mlp_offload_prefetch.py