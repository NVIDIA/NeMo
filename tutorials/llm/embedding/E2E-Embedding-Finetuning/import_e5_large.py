#!/usr/bin/env python3
"""
import_e5_large.py

This script downloads and converts the `intfloat/e5-large-v2` embedding model 
from Hugging Face into NeMo format using NVIDIA NeMo's `llm.import_ckpt` utility.

The resulting `.nemo` model file will be saved in the current working directory.
"""

import os
import logging
from nemo.collections import llm

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    # Step 1: Define working directory and environment paths
    cwd = os.getcwd()
    os.environ.setdefault("NEMO_HOME", cwd)
    os.environ["NEMO_MODELS_CACHE"] = cwd

    # Step 2: Create model config for E5-Large embeddings
    model_config = llm.BertEmbeddingModel(llm.BertEmbeddingLargeConfig())

    # Hugging Face source
    hf_source = 'hf://intfloat/e5-large-v2'

    # Step 3: Convert and save model
    output_file = os.path.join(cwd, 'e5-large-v2.nemo')
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

