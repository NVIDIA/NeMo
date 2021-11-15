#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
Updates a .nemo file with average weights.
"""

import os
import sys
import torch
import glob

from nemo.utils import logging, model_utils
from nemo.core import ModelPT


def main():
    device = torch.device("cpu")

    # loop over all folders with .nemo files (or .nemo files)
    for model_fname in sys.argv[1:]:
        if not model_fname.endswith(".nemo"):
            # assume model_fname is a folder which contains a .nemo file (filter .nemo files which matches with "averaged-*")
            nemo_files = list(filter(lambda fn: not os.path.basename(fn).startswith("averaged-"), glob.glob(os.path.join(model_fname, "*.nemo"))))
            if len(nemo_files) != 1:
                raise RuntimeError(f"Expected only a single .nemo files but discovered {len(nemo_files)} .nemo files")

            model_fname = nemo_files[0]

        model_folder_path = os.path.dirname(model_fname)
        avg_model_fname = os.path.join(model_folder_path, "averaged-"+os.path.basename(model_fname))

        logging.info(f"Parsing folder {model_folder_path}")

        # restore model from .nemo file path
        model_cfg = ModelPT.restore_from(restore_path=model_fname, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        nemo_model = imported_class.restore_from(restore_path=model_fname, map_location=device)  # type: ASRModel


        checkpoint_paths = [os.path.join(model_folder_path, x) for x in os.listdir(model_folder_path) if x.endswith('.ckpt')]
        """ < Checkpoint Averaging Logic > """
        # load state dicts
        n = len(checkpoint_paths)
        avg_state = None

        logging.info(f"Averaging {n} checkpoints ...")

        for ix, path in enumerate(checkpoint_paths):
            checkpoint = torch.load(path, map_location=device)

            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            if ix == 0:
                # Initial state
                avg_state = checkpoint

                logging.info(f"Initialized average state dict with checkpoint : {path}")
            else:
                # Accumulated state
                for k in avg_state:
                    avg_state[k] = avg_state[k] + checkpoint[k]

                logging.info(f"Updated average state dict with state from checkpoint : {path}")

        for k in avg_state:
            if str(avg_state[k].dtype).startswith("torch.int"):
                # For int type, not averaged, but only accumulated.
                # e.g. BatchNorm.num_batches_tracked
                pass
            else:
                avg_state[k] = avg_state[k] / n

        import pudb; pudb.set_trace()
        # Save model
        ckpt_name = os.path.join(model_folder_path, 'model_weights.ckpt')
        torch.save(avg_state, ckpt_name)
        logging.info(f"Averaged pytorch checkpoint saved as : {ckpt_name}")


if __name__ == '__main__':
    main()
