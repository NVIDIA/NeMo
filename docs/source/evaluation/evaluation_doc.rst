NeMo Evaluation User Guide
=======================

This guide provides details about how to evaluate NeMo 2.0 checkpoints with `lm-evaluation-harness
<https://github.com/EleutherAI/lm-evaluation-harness>`__ that is integrated with NeMo Framework.
Benchmarks supported: ``MMLU``, ``GSM8k``, ``lambada_openai``, ``winogrande``, ``arc_challenge``,
``arc_easy``, ``copa``

Introduction
--------------
Server client approach is used to run evaluations on NeMo 2.0 models. Running evaluation consists of 2 phases: 
Phase 1 is where the NeMo 2.0 checkpoint is deployed on PyTriton server by exporting it to TRTLLM. Phase 2 is where
evaluation is run on the model at the deployed url and port.

Running evaluations without NeMo-Run
--------------
This section covers the steps to deploy and evaluate NeMo 2.0 model without NeMo-Run using python commands directly.
This is a quick and easy approach to evaluate on a local workstation with GPUs as its easier to debug. 
However, to run on clusters its recommended to use NeMo-Run for the ease of use.
The entrypoint to deploy is the ``deploy`` method defined in ``nemo/collections/llm/api.py``. 
An example command to deploy is as below:

.. code-block:: python

    from nemo.collections.llm import deploy

    if __name__ == "__main__":
        deploy(
            nemo_checkpoint='/workspace/hf_llama3_8b_nemo2.nemo',
            max_input_len=4096,
            max_batch_size=4,
            num_gpus=1,)

The entrypoint for evaluation is the ``evaluate`` method defined in ``nemo/collections/llm/api.py``. In order to run
evaluations on the deployed model above, use the following command. Open a new terminal within the same container to 
run this. For evaluations taking longer, it's a good idea to run deploy and evaluate in tmux sessions to avoid the 
process from getting killed and aborting the runs.

.. code-block:: python

    from nemo.collections.llm import evaluate
    from nemo.collections.llm.evaluation.api import EvaluationConfig, ApiEndpoint, EvaluationTarget, ConfigParams

    nemo_checkpoint = '/workspace/hf_llama3_8b_nemo2.nemo/'
    api_endpoint = ApiEndpoint(nemo_checkpoint_path=nemo_checkpoint)
    eval_target = EvaluationTarget(api_endpoint=api_endpoint)
    eval_params = ConfigParams(top_p=1, temperature=1, top_k=1, limit_samples=2, num_fewshot=5)
    eval_config = EvaluationConfig(type='mmlu', params=eval_params)

    if __name__ == "__main__":
    evaluate(target_cfg=eval_target, eval_cfg=eval_config)

Note: Please refer to ``deploy`` and ``evaluate`` method in ``nemo/collections/llm/api.py`` to check all the argument 
options as these are just sample commands and don't share all arguments and their default settings.

Running evaluations with NeMo-Run
--------------

Note: For detailed information about `NeMo-Run <https://github.com/NVIDIA/NeMo-Run>`__, please refer to its
documentation. Below is a concise version that focuses on using NeMo-Run to run evaluations in NeMo 2.0.

Reference script to launch evaluations with NeMo-Run:
`evaluation.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/llm/evaluation.py>`__. This script provides example
for using NeMo-Run with both local executor (your local workstation) and slurm based executors like clusters. ``deploy``
and ``evaluate`` are launched as two separate jobs with NeMo-Run. The evaluate method waits until the PyTriton server
is accessible and the model is deployed before starting to run evaluations.

Example command to run locally with NeMo-Run:

.. code-block:: bash

    python scripts/llm/evaluation.py --nemo_checkpoint='/workspace/hf_llama3_8b_nemo2.nemo'

Note: With the local executor run, it is required to manually kill 

To run on slurm based clusters, please pass the ``--slurm`` flag to the command and add all custom parameters to the 
script like user, host, remote_job_dir, account, mounts etc., Please refer to the script for details. 
Example command below:

.. code-block:: bash

    python scripts/llm/evaluation.py --nemo_checkpoint='/workspace/hf_llama3_8b_nemo2.nemo' --slurm --nodes 1 
    --devices 8 --container_image "nvcr.io/nvidia/nemo:dev" --tensor_parallelism_size 8