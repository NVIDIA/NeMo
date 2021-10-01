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


from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from pytorch_lightning.trainer.trainer import Trainer
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=None,
        required=False,
        help="Path to PTL checkpoints saved during training.",
    )
    parser.add_argument("--nemo_file_path", type=str, default=None, required=False, help="Path to output .nemo file.")

    args = parser.parse_args()

    args.checkpoint_folder = '/raid/nemo_experiments/gpt_debug/megatron_gpt/2021-10-01_11-23-40/checkpoints/mp_rank_00/megatron_gpt--val_loss=8.79-step=49-last.ckpt'
    args.nemo_file_path = '~/tmp/ckpt_to_nemo.nemo'

    trainer = Trainer()
    model = MegatronGPTModel.load_from_checkpoint(checkpoint_path=args.checkpoint_folder, trainer=trainer)
    model.save_to(args.nemo_file_path)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
