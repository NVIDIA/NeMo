#!/usr/bin/env python3
"""
import_llama1b.py

This script downloads and converts a Hugging Face-hosted LLaMA-3 1B embedding model
into NeMo format using NVIDIA NeMo's `llm.import_ckpt` utility.

The final `.nemo` checkpoint is saved in the current working directory.
"""

import os
import logging
from nemo.collections import llm

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    # Step 1: Define working directory and set required environment variables
    cwd = os.getcwd()
    os.environ.setdefault("NEMO_HOME", cwd)
    os.environ["NEMO_MODELS_CACHE"] = cwd

    # Step 2: Create model config for LLaMA-3 1B embeddings
    model_config = llm.LlamaEmbeddingModel(llm.Llama32EmbeddingConfig1B())

    # Define Hugging Face source
    hf_source = 'hf://meta-llama/Llama-3.2-1B'

    # Step 3: Convert and save model to NeMo format
    output_file = os.path.join(cwd, 'Llama-3.2-1B.nemo')
    logging.info(f" Importing from {hf_source} → {output_file}...")

    llm.import_ckpt(
        model=model_config,
        source=hf_source,
        output_path=output_file,
    )

    # Step 4: Confirm success
    logging.info(f" Done. Checkpoint saved to {os.path.abspath(output_file)}")

if __name__ == '__main__':
    main()

