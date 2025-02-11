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

# pdb.set_trace = lambda: 1




def main(args) -> None:

    mbs = args.mbs
    gbs = args.gbs
    decoder_seq_length = 287
    # decoder_seq_length = 1024

    # Create the data module
    if True:
        from nemo.collections.vlm.clip.data.mock import MockDataModule as ClipMockDataModule
        # mock dataset
        data = ClipMockDataModule(
            seq_length=decoder_seq_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=None,
            num_train_samples=10_000_000_000,
            image_processor=None,
            num_workers=8,
        )
    else:
        data = OpenVLALazyDataModule(
        paths=args.data_path,
        data_mix=args.data_mix,
        llm_backbone_id="llama2-7b-pure",
        vision_backbone_id = "dinosiglip-vit-so-224px", # fused vision encoders
        # vision_backbone_id="clip-vit-l-336px",  # one vision encoder
        # vision_backbone_id = "siglip-vit-b16-224px", # one vision encoder
        shuffle_buffer_size=10,
        seq_length=decoder_seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=None,
        image_processor=None,
        num_workers=args.num_workers,
        hf_token=args.hf_token,
        load_for_training=False
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
        ckpt_load_strictness="log_all",
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            average_in_collective=True,
        ),
    )

    model = OpenVLAModel(OpenVLAHFConfig(), tokenizer=data.tokenizer)
    #
    # with torch.device("meta"):
    #
    #     concatenated_pixel_values = torch.randn(1, 6, 336, 336) * 0.9485338926315308 - 0.0002835348423104733
    #     input_ids = torch.tensor([[-200, 1, 512, 29901, 1724, 3158, 881, 278, 19964, 2125,
    #                                304, 5839, 701, 278, 13328, 18002, 29973, 13, 3744, 29901,
    #                                29871, 31999, 31872, 31872, 31872, 31872, 31872, 31744, 2, 0,
    #                                0, 0]])
    #     labels = torch.tensor([[-200, -100, -100, -100, -100, -100, -100, -100, -100, -100,
    #                             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
    #                             31999, 31872, 31872, 31872, 31872, 31872, 31744, 2, -100, -100,
    #                             -100, -100]])
    #     loss_mask = torch.tensor([[True, True, True, True, True, True, True, True, True, True,
    #                                True, True, True, True, True, True, True, True, True, True,
    #                                False, False, False, False, False, False, False, False, True, True,
    #                                True, True]])
    #     attention_mask = torch.tensor([[True, True, True, True, True, True, True, True, True, True,
    #                                     True, True, True, True, True, True, True, True, True, True,
    #                                     True, True, True, True, True, True, True, True, True, False,
    #                                     False, False]])
    #     position_ids = torch.arange(0, 32).unsqueeze(0)
    #
    #     # Construct the output dictionary
    #     output = dict(
    #         images=concatenated_pixel_values,
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #         loss_mask=loss_mask,
    #         position_ids=position_ids,
    #     )
    #
    # from lightning.pytorch.utilities import measure_flops
    # model_fwd = lambda: model(**output)
    # fwd_flops = measure_flops(model, model_fwd)
    #
    # model_loss = lambda y: y.sum()
    # fwd_and_bwd_flops = measure_flops(model, model_fwd, model_loss)


    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=10000,
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
        val_check_interval=10000,
        check_val_every_n_epoch=None,
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
        lr=2e-5,
        min_lr=2e-5,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        main_grads_dtype=torch.bfloat16,
        main_params_dtype=torch.bfloat16,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,  # exp_avg_sq_dtype
    )
    opt = MegatronOptimizerModule(config=opt_config)

    # import pdb; pdb.set_trace()
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
    parser.add_argument('--devices', type=int, default=8, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, default=50000, help="Number of steps to train for")
    parser.add_argument(
        '--experiment-dir', type=str, default=None, help="directory to write results and checkpoints to"
    )
    parser.add_argument('--hf-token', type=str, default=None, help="Huggingface access API token")
    parser.add_argument('--experiment-name', type=str, help="name of experiment")
    parser.add_argument('--wandb-project', type=str, default=None, help="wandb project name")

    parser.add_argument("--mbs", type=int, required=False, default=1, help="Micro batch size")
    # parser.add_argument("--mbs", type=int, required=False, default=32, help="Micro batch size")
    parser.add_argument("--gbs", type=int, required=False, default=1, help="Global batch size")
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--num_workers", type=int, required=False, default=8)
    parser.add_argument("mock_data", action="store_true")
    parser.add_argument(
        "--log_dir", type=str, required=False, default="/results", help="Directory for logging and checkpoints"
    )
    parser.add_argument("--wandb_project", type=str, required=False, default=None, help="Huggingface access API token")
    parser.add_argument("--name", type=str, required=False, default="openval-finetune")
    parser.add_argument("--restore_path", type=str, required=False,
                        default="/lustre/fsw/coreai_dlalgo_genai/abhgarg/openvla/openvla-converted-nemo2/openvla-7b")
    args = parser.parse_args()

    main(args)
