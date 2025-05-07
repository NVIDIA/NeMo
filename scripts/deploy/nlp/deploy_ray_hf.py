# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import logging
import signal
import sys

from nemo.deploy.deploy_ray import DeployRay
from nemo.deploy.nlp.hf_deployable_ray import HFRayDeployable

LOGGER = logging.getLogger("NeMo")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy a HuggingFace model using Ray")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model or model identifier from HuggingFace Hub",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-generation",
        help="HuggingFace task type (e.g., 'text-generation')",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading the model",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy for model placement",
    )
    parser.add_argument(
        "--tp_plan",
        type=str,
        default=None,
        help="Tensor parallelism plan for distributed inference",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="nemo-model",
        help="Identifier for the model in the API responses",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the Ray Serve server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1024,
        help="Port number to use for the Ray Serve server",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of CPUs to allocate for the Ray cluster",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to allocate for the Ray cluster",
    )
    parser.add_argument(
        "--include_dashboard",
        action="store_true",
        help="Whether to include the Ray dashboard",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="Number of model replicas to deploy",
    )
    parser.add_argument(
        "--num_gpus_per_replica",
        type=int,
        default=1,
        help="Number of GPUs per model replica",
    )
    parser.add_argument(
        "--num_cpus_per_replica",
        type=int,
        default=8,
        help="Number of CPUs per model replica",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0,1",
        help="Comma-separated list of CUDA visible devices",
    )
    return parser.parse_args()


def signal_handler(signum, frame, deployer):
    LOGGER.info("Received interrupt signal. Shutting down gracefully...")
    deployer.stop()
    sys.exit(0)


def main():
    args = parse_args()

    # Initialize Ray deployment
    ray_deployer = DeployRay(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        include_dashboard=args.include_dashboard,
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            }
        },
    )

    # Set up signal handlers
    signal.signal(signal.SIGINT, lambda signum, frame: signal_handler(signum, frame, ray_deployer))
    signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler(signum, frame, ray_deployer))

    try:
        # Start Ray Serve
        ray_deployer.start(host=args.host, port=args.port)

        # Create the HuggingFace model deployment
        app = HFRayDeployable.options(
            num_replicas=args.num_replicas,
            ray_actor_options={
                "num_gpus": args.num_gpus_per_replica,
                "num_cpus": args.num_cpus_per_replica,
            },
        ).bind(
            hf_model_id_path=args.model_path,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            device_map=args.device_map,
            tp_plan=args.tp_plan,
            model_id=args.model_id,
        )

        # Deploy the model
        ray_deployer.run(app, args.model_id)

        LOGGER.info(f"Model deployed successfully at {args.host}:{args.port}")
        LOGGER.info("Press Ctrl+C to stop the deployment")

        # Keep the script running
        while True:
            signal.pause()
    except Exception as e:
        LOGGER.error(f"Error during deployment: {str(e)}")
        ray_deployer.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
