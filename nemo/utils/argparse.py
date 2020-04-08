# Copyright (c) 2019 NVIDIA Corporation
import argparse
import os


class NemoArgParser(argparse.ArgumentParser):
    """
    This is a wrapper about argparse.

    Usage is the same as standard argparse.ArgParser:
        parser = NemoArgParser(description='Model_name')
        parser.add_argument(...)
        args = parser.parse_args()

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # NeMo arguments
        self.add_argument(
            "--local_rank", default=os.getenv('LOCAL_RANK', None), type=int, help="node rank for distributed training",
        )
        self.add_argument(
            "--amp_opt_level",
            default="O0",
            type=str,
            choices=["O0", "O1", "O2", "O3"],
            help="apex/amp precision:"
            " O0 - float32,"
            " O1 - mixed precision opt1,"
            " O2 - mixed precision opt2,"
            " O3 - float16"
            "See: https://nvidia.github.io/apex/amp.html",
        )
        self.add_argument(
            "--cudnn_benchmark",
            action="store_true",
            help="If set to True it will use cudnnFind method to"
            " find the best kernels instead of using "
            "heuristics. If the shapes of your inputs are "
            "constant this should help, for various shapes "
            "it can slow things down.",
        )
        # self.add_argument("--random_seed", default=None, type=int,
        #                   help="random seed")
        # self.add_argument("--deterministic", action="store_true",
        #                   help="whether to enable determinism")

        # Model defintion
        self.add_argument(
            "--model_config", type=str, default=None, help="model configuration file: model.yaml",
        )
        self.add_argument(
            "--train_dataset", type=str, default=None, help="training dataset path",
        )
        self.add_argument(
            "--eval_datasets", type=str, nargs="*", help="evaludation datasets paths",
        )
        self.add_argument("--batch_size", type=int, help="train batch size per GPU")
        self.add_argument(
            "--eval_batch_size", type=int, help="evaluation  batch size per GPU",
        )
        self.add_argument(
            "--eval_freq", default=1000, type=int, help="evaluation frequency, steps",
        )

        # Optimizer Choices
        self.add_argument(
            "--optimizer",
            type=str,
            choices=["sgd", "adam", "fused_adam", "adam_w", "novograd", "lamb",],
            help="optimizer",
        )
        self.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
        # self.add_argument("--momentum", type=float,
        #                   help="SGD momentum")
        # self.add_argument("--beta1", type=float,
        #                   help="Adam/AdamW/NovoGrad beta1")
        # self.add_argument("--beta2", type=float,
        #                   help="Adam/AdamW/NovoGrad beta2")

        # Optimization Arguments
        self.add_argument(
            "--num_epochs",
            type=int,
            default=None,
            help="number of epochs to train. You should specify either num_epochs or max_steps",
        )
        self.add_argument(
            "--max_steps",
            type=int,
            default=None,
            help="max number of steps to train. You should specify either num_epochs or max_steps",
        )
        self.add_argument("--lr", type=float, default=1e-3, help="base learning rate")
        self.add_argument(
            "--lr_policy", type=str, default='WarmupAnnealing', help="learning rate decay policy",
        )
        # self.add_argument("--warmup_steps", default=0, type=int,
        #                   help="number of learning rate warmup steps")
        self.add_argument(
            "--iter_per_step",
            default=1,
            type=int,
            help="number of gradients accumulation iterations per weights update step",
        )

        # Logging arguments
        self.add_argument(
            "--work_dir", default=None, type=str, help="working directory for experiment",
        )
        self.add_argument(
            "--checkpoint_dir",
            default=None,
            type=str,
            help="where to save checkpoints. If ckpt_dir is "
            "None, the default behaviour is to put it under"
            "{work_dir}/checkpoints",
        )
        self.add_argument(
            "--create_tb_writer", action="store_true", help="whether to log into Tensorboard",
        )
        self.add_argument(
            "--tensorboard_dir",
            default=None,
            type=str,
            help="If --create_tb_writer is enabled, specifies "
            "the tensorboard directory. Defaults to "
            "{work_dir}/checkpoints",
        )
        self.add_argument(
            "--checkpoint_save_freq", default=1000, type=int, help="checkpoint frequency, steps",
        )
