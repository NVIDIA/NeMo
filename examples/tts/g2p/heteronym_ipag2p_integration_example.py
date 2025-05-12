#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates the proper integration of HeteronymClassification with IpaG2p for TTS.
It shows how to initialize the heteronym model and use it with IpaG2p in a multiprocessing context.
"""

import os
import argparse
import torch
import multiprocessing
from typing import List
from torch.multiprocessing import Pool

from nemo.collections.tts.g2p.models.heteronym_classification import HeteronymClassificationModel
from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p
from nemo.utils import logging


def process_text(ipa_g2p: IpaG2p, text: str) -> List[str]:
    """Process a single text with IpaG2p in a worker process."""
    logging.info(f"Processing text in process {os.getpid()}: {text[:30]}...")
    return ipa_g2p(text)


def process_batch(args, batch: List[str]):
    """Process a batch of text in separate processes."""
    
    # Initialize IpaG2p with heteronym model in the main process
    heteronym_model = None
    if args.heteronym_model_path:
        logging.info(f"Loading heteronym model from {args.heteronym_model_path}")
        heteronym_model = HeteronymClassificationModel.restore_from(args.heteronym_model_path)
    
    # Initialize IpaG2p
    ipa_g2p = IpaG2p(
        phoneme_dict=args.phoneme_dict,
        ignore_ambiguous_words=not args.use_ambiguous_words,
        heteronyms=args.heteronyms_file,
    )
    
    # Setup heteronym model if available
    if heteronym_model:
        ipa_g2p.setup_heteronym_model(
            heteronym_model=heteronym_model,
            wordid_to_phonemes_file=args.wordid_to_phonemes_file
        )
    
    # Process texts in separate processes
    # The HeteronymModelSingleton ensures that the model is properly handled in multiprocessing
    with Pool(processes=args.num_workers) as pool:
        results = pool.starmap(
            process_text,
            [(ipa_g2p, text) for text in batch]
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Example of integrating HeteronymClassification with IpaG2p in a multiprocessing context"
    )
    parser.add_argument(
        "--phoneme_dict", 
        required=True, 
        help="Path to the IPA pronunciation dictionary file"
    )
    parser.add_argument(
        "--heteronym_model_path", 
        default=None, 
        help="Path to the pretrained HeteronymClassificationModel .nemo file"
    )
    parser.add_argument(
        "--wordid_to_phonemes_file", 
        default=None, 
        help="Path to wordid to phonemes mapping file for heteronym disambiguation"
    )
    parser.add_argument(
        "--heteronyms_file", 
        default=None, 
        help="Path to a file containing list of heteronyms"
    )
    parser.add_argument(
        "--input_file", 
        default=None, 
        help="Path to a text file with sentences to process (one per line)"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=2, 
        help="Number of worker processes"
    )
    parser.add_argument(
        "--use_ambiguous_words", 
        action="store_true", 
        help="Don't ignore ambiguous words in G2P"
    )
    
    args = parser.parse_args()
    
    # Ensure 'spawn' method for multiprocessing when using CUDA
    # Note: This is now handled automatically by the HeteronymModelSingleton,
    # but it's good practice to make it explicit here as well
    if torch.cuda.is_available():
        try:
            multiprocessing.get_start_method()
        except RuntimeError:
            multiprocessing.set_start_method('spawn', force=True)
        logging.info("Using 'spawn' method for multiprocessing with CUDA")
    
    # Sample texts for demonstration
    sample_texts = [
        "Reading the novel was easy.",
        "I was reading a book about farming and produce.",
        "The tears in her eyes were visible.",
        "The fabric tears easily.",
        "The wind was very strong yesterday.",
        "You need to wind the clock."
    ]
    
    # Load texts from file if provided
    if args.input_file and os.path.exists(args.input_file):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sample_texts = [line.strip() for line in f if line.strip()]
    
    # Process texts
    results = process_batch(args, sample_texts)
    
    # Print results
    for i, (text, phonemes) in enumerate(zip(sample_texts, results)):
        logging.info(f"Text {i+1}: {text}")
        logging.info(f"Phonemes: {phonemes}")
        logging.info("-" * 50)


if __name__ == "__main__":
    main()
