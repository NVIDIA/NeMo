Fast Training 
=============

Training a large model (especially from scratch) requires significant compute. NeMo provides support for mixed precision and distributed training to speed-up training. NeMo uses `NVIDIA's APEX library <https://github.com/NVIDIA/apex>`_ to get maximum performance out of NVIDIA's GPUs. Furthermore, multi-GPU systems (such as DGX Station, DGX-1 and DGX-2) have *NVLINK* to speed-up multi-GPU communication.


Mixed Precision
~~~~~~~~~~~~~~~
NVIDIA Volta and Turing GPUs have *Tensor Cores* which can do fast matrix multiplications with values in float16 format.
To enable mixed-precision in NeMo all you need to do is to set `optimization_level` parameter of `nemo.core.NeuralModuleFactory` to `nemo.core.Optimization.mxprO1`. For example:

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
           optimization_level=nemo.core.Optimization.mxprO1)

.. important::
    Mixed precision requires Tensor Cores, so it works only on NVIDIA Volta and Turing GPUs.

Multi-GPU training
~~~~~~~~~~~~~~~~~~

For multi-GPU training:

(1) Set `placement` to `nemo.core.DeviceType.AllGpu` in NeuralModuleFactory
(2) Add 'local_rank' argument to your script and do not set it yourself: `parser.add_argument("--local_rank", default=None, type=int)`

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
           placement=nemo.core.DeviceType.AllGpu,     
           local_rank=args.local_rank)


Use `torch.distributed.launch` package to run your script like this (assuming 8 GPUs):

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=8 <nemo_git_repo_root>/examples/asr/jasper.py --num_gpus=8 ...


Example
~~~~~~~

Please refer to the `<nemo_git_repo_root>/examples/asr/jasper.py` for a comprehensive example. 
It builds one train DAG and up to three validation DAGs to evaluate on different datasets.

If you are working with a Volta-based DGX, you can run training like this:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=8 <nemo_git_repo_root>/examples/asr/jasper.py --batch_size=64 --num_gpus=8 --num_epochs=100 --lr=0.015 --warmup_steps=8000 --weight_decay=0.001 --train_manifest=/manifests/librivox-train-all.json --val_manifest1=/manifests/librivox-dev-clean.json --val_manifest2=/manifests/librivox-dev-other.json --model_config=<nemo_git_repo_root>/nemo/examples/asr/configs/jasper15x5SEP.yaml --exp_name=MyLARGE-ASR-EXPERIMENT

The command above should trigger 8-GPU training with mixed precision. In the command above various manifests (.json) files are various datasets. Substitute them with the ones containing your data.

.. tip::
    You can pass several manifests (comma-separated) to train on a combined dataset like this: `--train_manifest=/manifests/librivox-train-all.json,/manifests/librivox-train-all-sp10pcnt.json,/manifests/cv/validated.json`. 

This example would train on 3 data sets: LibriSpeech, Mozilla Common Voice and LibriSpeech speed perturbed.


