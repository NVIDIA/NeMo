# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
Usage example:
    wget https://upload.wikimedia.org/wikipedia/commons/0/0f/1665_Girl_with_a_Pearl_Earring.jpg
    python scripts/vlm/clip_infer.py --image_url 1665_Girl_with_a_Pearl_Earring.jpg \
    --hf_path hf://openai/clip-vit-large-patch14 \
    --classes "a dog" "a boy" "a girl"



It should generate a high probability for "a girl" tag, e.g.
Nemo: CLIP text probability:  [('a dog', 0.0051774755), ('a boy', 0.0024592995), ('a girl', 0.9923632)]
HF: CLIP text probability:  [('a dog', 0.004963576), ('a boy', 0.0022506083), ('a girl', 0.9927858)]
"""
import argparse
import os

import requests
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger
# from torchvision.transforms import ToTensor
import nemo.lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.vla.openvla.base import OpenVLAModel
from nemo.collections.vla.openvla.openvla import OpenVLAHFConfig


import pdb, torch

from nemo.collections.vlm.openvla.data.lazy import OpenVLALazyDataModule
# from nemo.collections.vlm.openvla.data.lazy import OpenVLALazyDataModule
from nemo.lightning import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

pdb.set_trace = lambda: 1




def main(args) -> None:

    mbs = args.mbs
    gbs = args.mbs
    decoder_seq_length = 287
    decoder_seq_length = 287

    # Create the data module
    if False:
        # mock dataset
        data = vlm.ClipMockDataModule(
            seq_length=decoder_seq_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=None,
            num_train_samples=10_000_000_000,
            image_processor=None,
            num_workers=1,
        )
    else:
        data = OpenVLALazyDataModule(
        paths=args.data_path,
        data_mix=args.data_mix,
        llm_backbone_id="llama2-7b-pure",
        vision_backbone_id = "dinosiglip-vit-so-224px", # fused vision encoders
        # vision_backbone_id="clip-vit-l-336px",  # one vision encoder
        # vision_backbone_id = "siglip-vit-b16-224px", # one vision encoder
        shuffle_buffer_size=100,
        seq_length=decoder_seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=None,
        image_processor=None,
        num_workers=4,
        hf_token=args.hf_token,
    )

    # pylint: disable=C0115,C0116
    strategy = nl.MegatronStrategy(

    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
        ckpt_load_strictness="log_all",
    )


    model = OpenVLAModel(OpenVLAHFConfig(), tokenizer=data.tokenizer)

    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=1000,
        dirpath=os.path.join(args.log_dir, args.name),
    )


    trainer = nl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=100,
        limit_val_batches=1,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_from_directory=os.path.join(args.log_dir, args.name),
        restore_config=nl.RestoreConfig(path=args.restore_path) if args.restore_path is not None else None,
    )


    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        min_lr=6e-5,
        use_distributed_optimizer=False,
        bf16=True,
    )
    opt = MegatronOptimizerModule(config=opt_config)

    import pdb; pdb.set_trace()
    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )

    llm.finetune(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume,
        optim=opt,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip Verification Script")

    parser.add_argument(
        "--hf_path",
        type=str,
        default="hf://openvla/openvla-7b",
        help="Path to the Huggingface model.",
    )

    parser.add_argument('--data-path', type=str, default=None, help="Path to dataset collection")
    parser.add_argument('--data-mix', type=str, default=None, help="Data mix selected")
    parser.add_argument('--devices', type=int, default=1, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, default=5, help="Number of steps to train for")
    parser.add_argument(
        '--experiment-dir', type=str, default=None, help="directory to write results and checkpoints to"
    )
    parser.add_argument('--hf-token', type=str, default=None, help="Huggingface access API token")
    parser.add_argument('--experiment-name', type=str, help="name of experiment")
    parser.add_argument('--wandb-project', type=str, default=None, help="wandb project name")

    parser.add_argument("--mbs", type=int, required=False, default=32, help="Micro batch size")
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--num_workers", type=int, required=False, default=8)
    parser.add_argument("mock_data", action="store_true")
    parser.add_argument(
        "--log_dir", type=str, required=False, default="/results", help="Directory for logging and checkpoints"
    )
    parser.add_argument("--name", type=str, required=False, default="openval-finetune")
    parser.add_argument("--restore_path", type=str, required=False,
                        default="/lustre/fsw/coreai_dlalgo_genai/abhgarg/openvla/openvla-converted-nemo2/openvla-7b")
    args = parser.parse_args()

    main(args)
