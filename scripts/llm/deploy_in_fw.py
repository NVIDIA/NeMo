import argparse
from nemo.collections.llm import deploy


def get_parser():
    parser = argparse.ArgumentParser(description="NeMo2.0 Pretraining")
    parser.add_argument(
        "--nemo_checkpoint",
        type=str,
        help="NeMo 2.0 checkpoint to be evaluated",
    ),
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Num of gpus per node",
    ),
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Num of nodes",
    ),
    parser.add_argument(
        "--tensor_parallelism_size",
        type=int,
        default=1,
        help="Tensor parallelism size to deploy the model",
    ),
    parser.add_argument(
        "--pipeline_parallelism_size",
        type=int,
        default=1,
        help="Pipeline parallelism size to deploy the model",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="context parallelism size to deploy the model",
    )
    parser.add_argument(
        "--expert_model_parallel_size",
        type=int,
        default=1,
        help="Expert model parallelism size to deploy the model",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    deploy(
        nemo_checkpoint=args.nemo_checkpoint,
        num_gpus=args.ngpus,
        num_nodes=args.nnodes,
        fastapi_port=8886,
        tensor_parallelism_size=args.tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
    )
