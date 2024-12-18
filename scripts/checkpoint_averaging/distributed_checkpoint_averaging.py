# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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
Example: python scripts/checkpoint_averaging/distributed_checkpoint_averaging.py \
             --name_prefix=<checkpoint name> \
             --checkpoint_dir=<folder with mp_rank_X subfolders containing checkpoints>
             --steps <optinally a list of checkpoint steps to average, if not provided, it will average all the checkpoints>

will generate a new directory in each of the distributed checkpoint subfolders named <checkpoint name>-averaged
"""

import argparse
import logging
import os
import shutil

import numpy as np
import tensorstore  # need to import it for bf16 support
import torch
import zarr
from lightning.pytorch.trainer.trainer import Trainer

logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name_prefix',
        help='Name of the final checkpoint. Will append -averaged automatically.',
    )
    parser.add_argument(
        '--checkpoint_dir',
        help='Folder containing all the distributed checkpoints.',
    )
    parser.add_argument(
        '--checkpoint_format',
        default='torch_dist',
        choices=['torch_dist', 'zarr'],
        help='Format of distributed checkpoint.',
    )
    parser.add_argument('--hparams_file', help='Path to hparams.yaml.')
    parser.add_argument('--precision', default='bf16-mixed', help='Model precision.')
    # list of checkpoint steps to average
    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        help='List of checkpoint steps to average. If not specified, will average all.',
    )

    args = parser.parse_args()

    return args


def init_trainer(args):
    from lightning.pytorch.trainer.trainer import Trainer
    from megatron.core import parallel_state
    from omegaconf import OmegaConf, open_dict

    from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
    from nemo.collections.nlp.parts.nlp_overrides import (
        GradScaler,
        NLPDDPStrategy,
        NLPSaveRestoreConnector,
        PipelineMixedPrecisionPlugin,
    )

    plugins = []

    cfg = {
        'trainer': {
            'accelerator': 'cpu',
            'precision': args.precision,
        },
        'model': {
            'native_amp_init_scale': 2**32,
            'native_amp_growth_interval': 1000,
            'hysteresis': 2,
            'gradient_as_bucket_view': True,
        },
    }
    cfg = OmegaConf.create(cfg)

    scaler = None
    # If FP16 create a GradScaler as the build_model_parallel_config of MegatronBaseModel expects it
    if cfg.trainer.precision == '16-mixed':
        scaler = GradScaler(
            init_scale=cfg.model.get('native_amp_init_scale', 2**32),
            growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
            hysteresis=cfg.model.get('hysteresis', 2),
        )
    plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cpu', scaler=scaler))
    # Set precision None after precision plugins are created as PTL >= 2.1 does not allow both
    # precision plugins and precision to exist
    strategy = NLPDDPStrategy()
    cfg.trainer.precision = None
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    return trainer


def load_torch_dist_ckpt(path, hparams_file, trainer, return_ckpt=False):
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

    model = MegatronGPTModel.load_from_checkpoint(
        checkpoint_path=path, hparams_file=hparams_file, trainer=trainer, return_ckpt=return_ckpt
    )

    return model


def main(args):
    if args.checkpoint_format == 'torch_dist':

        # local_rank, rank, world_size = initialize_distributed(args)
        trainer = init_trainer(args)  # , world_size)

    if args.steps is not None:
        logging.info(f"Will average only steps {args.steps}")

    # repeating for all ranks

    checkpoint_paths = []
    for ckpt_dir in os.listdir(args.checkpoint_dir):
        logging.info("Processing %s", ckpt_dir)
        if ckpt_dir.endswith('0-last'):
            continue
        if args.steps is None:
            checkpoint_paths.append(ckpt_dir)
        else:
            for step in args.steps:
                key = f"-step={step}-"
                if key in ckpt_dir:
                    checkpoint_paths.append(ckpt_dir)

    n = len(checkpoint_paths)
    # initialize dict, will be used to store the weights that need to be averaged
    avg_weights = {}
    chunk_info = {}

    logging.info(f"Averaging {n} checkpoints ... {'at steps:' + str(args.steps) if args.steps is not None else ''}")

    # item that needs to be copied to the new checkpoint folder
    copy_items = []
    for ix, path in enumerate(checkpoint_paths):
        full_path = os.path.join(args.checkpoint_dir, path)

        if args.checkpoint_format == 'zarr':
            for item in os.listdir(full_path):

                # if item is not a directory, skip it
                if not os.path.isdir(os.path.join(full_path, item)):
                    if ix == 0:
                        copy_items.append(os.path.join(full_path, item))
                    continue

                # transformer engine states, leave them out
                if item.endswith('._extra_state'):
                    if ix == 0:
                        copy_items.append(os.path.join(full_path, item))
                    continue

                # optimizer states, no point of averaing them
                if item.startswith('optimizer.'):
                    if ix == 0:
                        copy_items.append(os.path.join(full_path, item))
                    continue

                if item not in avg_weights:
                    logging.info(f"Initialized average weights dict with: {item}")
                    array = zarr.open(os.path.join(full_path, item), mode='r')
                    avg_weights[item] = array[:]
                    chunk_info[item] = array.chunks
                else:
                    logging.info(f"Updated average weights dict with weight: {item}")
                    array_z = zarr.open(os.path.join(full_path, item), mode='r')
                    sum_array = avg_weights[item] + array_z[:]
                    avg_weights[item] = sum_array
        else:
            model = load_torch_dist_ckpt(full_path, args.hparams_file, trainer)
            for key, value in model.state_dict().items():
                if "_extra_state" not in key:
                    if key not in avg_weights:
                        avg_weights[key] = value.to('cpu')
                    else:
                        avg_weights[key] += value.to('cpu')

    for k in avg_weights:
        logging.info(f"Average weights dict key : {k}, dtype : {avg_weights[k].dtype}, shape : {avg_weights[k].shape}")
        if str(avg_weights[k].dtype).startswith("int"):
            raise ValueError("Int type not supported")
        else:
            array_z = avg_weights[k] / n
            avg_weights[k] = array_z

    if args.checkpoint_format == 'zarr':
        # Save model
        if args.steps is None:
            ckpt_name = os.path.join(args.checkpoint_dir, args.name_prefix + '-averaged')
        else:
            steps_combined = '_'.join([str(x) for x in args.steps])
            ckpt_name = os.path.join(args.checkpoint_dir, args.name_prefix + '-' + steps_combined + '-averaged')

        # save avg_weights
        for k in avg_weights:
            logging.info(f"Saving {k} to {ckpt_name}")
            input_arr = avg_weights[k]
            chunks = chunk_info[k]
            # create the zarr array
            output_array = zarr.create(
                input_arr.shape,
                dtype=input_arr.dtype,
                store=os.path.join(ckpt_name, k),
                chunks=chunks,
                compressor=None,
                fill_value=None,
                write_empty_chunks=True,
            )
            if input_arr.dtype == np.dtype('bfloat16'):
                arr = output_array
                arr._dtype = input_arr.dtype
                zarray = arr.store['.zarray']
                arr.store['.zarray'] = zarray.replace(b'<V2', b'bfloat16')
            output_array[:] = input_arr

        # copy other files
        for item in copy_items:
            is_file = os.path.isfile(item)
            logging.info(f"Copying {'directory' if is_file else 'file'} {item} to {ckpt_name}")
            if os.path.isfile(item):
                # copy single file
                shutil.copy(item, ckpt_name)
            else:
                # copy directory
                shutil.copytree(item, os.path.join(ckpt_name, os.path.basename(item)), dirs_exist_ok=True)
        logging.info(f"Averaged distributed checkpoint saved as : {ckpt_name}")
    else:
        avg_model_path = os.path.join(args.checkpoint_dir, checkpoint_paths[0])
        avg_model = load_torch_dist_ckpt(avg_model_path, args.hparams_file, trainer, return_ckpt=False)
        # avg_state_dict = avg_model.state_dict()
        # for key, value in avg_weights.items():
        #     avg_state_dict[key] = value

        from megatron.core import dist_checkpointing

        # avg_model['state_dict'] = avg_state_dict
        os.mkdir(os.path.join(args.checkpoint_dir, "average"))
        from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

        # strat = NLPDDPStrategy()
        avg_model.save_to(os.path.join(f"{args.checkpoint_dir}/avg", "average.nemo"))
        # strat.save_checkpoint(checkpoint=avg_model, os.path.join(args.checkpoint_dir, "average"))


if __name__ == '__main__':

    args = get_args()
    main(args)
