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
Module that provides a singleton pattern for HeteronymClassificationModel to ensure
it's properly initialized for multiprocessing contexts.
"""

import multiprocessing
import threading
from typing import List, Optional

from nemo.utils import logging


class HeteronymModelSingleton:
    """
    Singleton class for HeteronymClassificationModel to ensure proper initialization
    in multiprocessing contexts.
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    @classmethod
    def get_instance(cls, model=None, wordid_to_phonemes_file: Optional[str] = None):
        """
        Get or create the singleton instance of the HeteronymClassificationModel.

        Args:
            model: The HeteronymClassificationModel instance to use.
            wordid_to_phonemes_file: Optional path to the wordid to phonemes mapping file.

        Returns:
            The singleton instance of the HeteronymClassificationModel.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Set multiprocessing start method to 'spawn' to avoid CUDA issues
                    try:
                        # Only set if it hasn't been set already
                        multiprocessing.get_start_method()
                    except RuntimeError:
                        multiprocessing.set_start_method('spawn', force=True)
                        logging.info("Multiprocessing start method set to 'spawn' for HeteronymClassificationModel")

                    if model is not None:
                        cls._instance = model
                        cls._initialized = True

                        # Set wordid_to_phonemes if provided
                        if wordid_to_phonemes_file is not None:
                            cls._instance.set_wordid_to_phonemes(wordid_to_phonemes_file)

                        logging.info("HeteronymClassificationModel singleton initialized")
                    else:
                        logging.warning("No model provided to HeteronymModelSingleton")

        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (primarily for testing)."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False

    @classmethod
    def disambiguate(cls, sentences: List[str], batch_size: int = 4, num_workers: int = 0) -> tuple:
        """
        Wrapper for the HeteronymClassificationModel's disambiguate method.

        Args:
            sentences: List of sentences to disambiguate.
            batch_size: Batch size for inference.
            num_workers: Number of workers for data loading.

        Returns:
            Tuple of (input_texts, output_texts, all_predictions) from the model.
            If no model is initialized, returns the input sentences unchanged.
        """
        instance = cls.get_instance()

        if instance is not None and cls._initialized:
            try:
                return instance.disambiguate(sentences=sentences, batch_size=batch_size, num_workers=num_workers)
            except Exception as e:
                logging.warning(f"HeteronymClassificationModel disambiguation failed: {e}")
                # Return the original sentences as fallback
                return sentences, sentences, [[] for _ in sentences]
        else:
            # If no model is available, return the input unchanged
            return sentences, sentences, [[] for _ in sentences]
