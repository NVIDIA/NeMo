Tutorial
========

Make sure you have installed ``nemo`` and ``nemo_asr`` collection.
See :ref:`installation` section.

.. note::
    You only need `nemo` and `nemo_asr` collection for this tutorial.

Introduction
-------------

This Automatic Speech Recognition (ASR) tutorial is focused on Jasper :cite:`li2019jasper` model. Jasper is CTC-based :cite:`graves2006` end-to-end model. The model is called "end-to-end" because it transcripts speech samples without any additional alignment information. CTC allows finding an alignment between audio and text. 
CTC-ASR training pipeline consists of the following blocks:

1. audio preprocessing (feature extraction): signal normalization, windowing, (log) spectrogram (or mel scale spectrogram, or MFCC)
2. neural acoustic model (which predicts a probability distribution P_t(c) over vocabulary characters c per each time step t given input features per each timestep)
3. CTC loss function

    .. image:: ctc_asr.png
        :align: center
        :alt: CTC-based ASR



Get data
--------
We will be using an open-source LibriSpeech :cite:`panayotov2015librispeech` dataset. These scripts will download and convert LibriSpeech into format expected by `nemo_asr`:

.. code-block:: bash

    mkdir data
    # note that this script requires sox to be installed
    # to install sox on Ubuntu, simply do: sudo apt-get install sox
    # and then: pip install sox
    # get_librispeech_data.py script is located under <nemo_git_repo_root>/scripts
    python get_librispeech_data.py --data_root=data --data_set=dev_clean,train_clean_100
    # To get all LibriSpeech data, do:
    # python get_librispeech_data.py --data_root=data --data_set=ALL

.. note::
    You should have at least 26GB of disk space available if you've used ``--data_set=dev_clean,train_clean_100``; and at least 110GB if you used ``--data_set=ALL``. Also, it will take some time to download and process, so go grab a coffee.


After download and conversion, your `data` folder should contain 2 json files:

* dev_clean.json
* train_clean_100.json

In the tutorial we will use `train_clean_100.json` for training and `dev_clean.json`for evaluation.
Each line in json file describes a training sample - `audio_filepath` contains path to the wav file, `duration` it's duration in seconds, and `text` is it's transcript:

.. code-block:: json

    {"audio_filepath": "<absolute_path_to>/1355-39947-0000.wav", "duration": 11.3, "text": "psychotherapy and the community both the physician and the patient find their place in the community the life interests of which are superior to the interests of the individual"}
    {"audio_filepath": "<absolute_path_to>/1355-39947-0001.wav", "duration": 15.905, "text": "it is an unavoidable question how far from the higher point of view of the social mind the psychotherapeutic efforts should be encouraged or suppressed are there any conditions which suggest suspicion of or direct opposition to such curative work"}



Training 
---------

We will train a small model from the Jasper family :cite:`li2019jasper`.
Jasper ("Just Another SPeech Recognizer") is a deep time delay neural network (TDNN) comprising of blocks of 1D-convolutional layers. 
Jasper family of models are denoted as Jasper_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D convolution, batch normalization, ReLU, and dropout:

    .. image:: jasper.png
        :align: center
        :alt: japer model


In the tutorial we will be using model [12x1] and will be using separable convolutions.
The script below does both training (on `train_clean_100.json`) and evaluation (on `dev_clean.json`) on single GPU:

    .. tip::
        Run Jupyter notebook and walk through this script step-by-step


**Training script**

.. code-block:: python

    # NeMo's "core" package
    import nemo
    # NeMo's ASR collection
    import nemo_asr

    # Create a Neural Factory
    # It creates log files and tensorboard writers for us among other functions
    nf = nemo.core.NeuralModuleFactory(
        log_dir='jasper12x1SEP',
        create_tb_writer=True)
    tb_writer = nf.tb_writer
    logger = nf.logger

    # Path to our training manifest
    train_dataset = "<path_to_where_you_put_data>/train_clean_100.json"

    # Path to our validation manifest
    eval_datasets = "<path_to_where_you_put_data>/dev_clean.json"

    # Jasper Model definition
    from ruamel.yaml import YAML

    # Here we will be using separable convolutions
    # with 12 blocks (k=12 repeated once r=1 from the picture above)
    yaml = YAML(typ="safe")
    with open("<nemo_git_repo_root>/examples/asr/configs/jasper12x1SEP.yaml") as f:
        jasper_model_definition = yaml.load(f)
    labels = jasper_model_definition['labels']

    # Instantiate neural modules
    data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=train_dataset,
        labels=labels, batch_size=32)
    data_layer_val = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=eval_datasets,
        labels=labels, batch_size=32, shuffle=False)

    data_preprocessor = nemo_asr.AudioPreprocessing()
    spec_augment = nemo_asr.SpectrogramAugmentation(rect_masks=5)

    jasper_encoder = nemo_asr.JasperEncoder(
        feat_in=64,
        **jasper_model_definition['JasperEncoder'])
    jasper_decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=1024, num_classes=len(labels))
    ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    # Training DAG (Model)
    audio_signal, audio_signal_len, transcript, transcript_len = data_layer()
    processed_signal, processed_signal_len = data_preprocessor(
        input_signal=audio_signal, length=audio_signal_len)
    aug_signal = spec_augment(input_spec=processed_signal)
    encoded, encoded_len = jasper_encoder(
        audio_signal=aug_signal, length=processed_signal_len)
    log_probs = jasper_decoder(encoder_output=encoded)
    predictions = greedy_decoder(log_probs=log_probs)
    loss = ctc_loss(
        log_probs=log_probs, targets=transcript,
        input_length=encoded_len, target_length=transcript_len)

    # Validation DAG (Model)
    # We need to instantiate additional data layer neural module
    # for validation data
    audio_signal_v, audio_signal_len_v, transcript_v, transcript_len_v = data_layer_val()
    processed_signal_v, processed_signal_len_v = data_preprocessor(
        input_signal=audio_signal_v, length=audio_signal_len_v)
    # Note that we are not using data-augmentation in validation DAG
    encoded_v, encoded_len_v = jasper_encoder(
        audio_signal=processed_signal_v, length=processed_signal_len_v)
    log_probs_v = jasper_decoder(encoder_output=encoded_v)
    predictions_v = greedy_decoder(log_probs=log_probs_v)
    loss_v = ctc_loss(
        log_probs=log_probs_v, targets=transcript_v,
        input_length=encoded_len_v, target_length=transcript_len_v)

    # These helper functions are needed to print and compute various metrics
    # such as word error rate and log them into tensorboard
    # they are domain-specific and are provided by NeMo's collections
    from nemo_asr.helpers import monitor_asr_train_progress, \
        process_evaluation_batch, process_evaluation_epoch

    from functools import partial
    # Callback to track loss and print predictions during training
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tb_writer=tb_writer,
        # Define the tensors that you want SimpleLossLoggerCallback to
        # operate on
        # Here we want to print our loss, and our word error rate which
        # is a function of our predictions, transcript, and transcript_len
        tensors=[loss, predictions, transcript, transcript_len],
        # To print logs to screen, define a print_func
        print_func=partial(
            monitor_asr_train_progress,
            labels=labels,
            logger=logger
        ))

    saver_callback = nemo.core.CheckpointCallback(
        folder="./",
        # Set how often we want to save checkpoints
        step_freq=100)

    # PRO TIP: while you can only have 1 train DAG, you can have as many
    # val DAGs and callbacks as you want. This is useful if you want to monitor
    # progress on more than one val dataset at once (say LibriSpeech dev clean
    # and dev other)
    eval_callback = nemo.core.EvaluatorCallback(
        eval_tensors=[loss_v, predictions_v, transcript_v, transcript_len_v],
        # how to process evaluation batch - e.g. compute WER
        user_iter_callback=partial(
            process_evaluation_batch,
            labels=labels
            ),
        # how to aggregate statistics (e.g. WER) for the evaluation epoch
        user_epochs_done_callback=partial(
            process_evaluation_epoch, tag="DEV-CLEAN", logger=logger
            ),
        eval_step=500,
        tb_writer=tb_writer)

    # Run training using your Neural Factory
    # Once this "action" is called data starts flowing along train and eval DAGs
    # and computations start to happen
    nf.train(
        # Specify the loss to optimize for
        tensors_to_optimize=[loss],
        # Specify which callbacks you want to run
        callbacks=[train_callback, eval_callback, saver_callback],
        # Specify what optimizer to use
        optimizer="novograd",
        # Specify optimizer parameters such as num_epochs and lr
        optimization_params={
            "num_epochs": 50, "lr": 0.02, "weight_decay": 1e-4
            }
        )

.. note::
    This script trains should finish 50 epochs in about 7 hours on GTX 1080.

.. tip::
    To improve your word error rates:
        (1) Train longer
        (2) Train on more data
        (3) Use larger model
        (4) Train on several GPUs and use mixed precision (on NVIDIA Volta and Turing GPUs)
        (5) Start with pre-trained checkpoints


Mixed Precision training
-------------------------
Mixed precision and distributed training in NeMo is based on `NVIDIA's APEX library <https://github.com/NVIDIA/apex>`_.
Make sure it is installed.

To train with mixed-precision all you need is to set `optimization_level` parameter of `nemo.core.NeuralModuleFactory`  to `nemo.core.Optimization.mxprO1`. For example:

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=nemo.core.Optimization.mxprO1,
        placement=nemo.core.DeviceType.AllGpu,
        cudnn_benchmark=True)

.. note::
    Because mixed precision requires Tensor Cores it only works on NVIDIA Volta and Turing based GPUs

Multi-GPU training
-------------------

Enabling multi-GPU training with NeMo is easy:

   (1) First set `placement` to `nemo.core.DeviceType.AllGpu` in NeuralModuleFactory and in your Neural Modules
   (2) Have your script accept 'local_rank' argument and do not set it yourself: `parser.add_argument("--local_rank", default=None, type=int)`
   (3) Use `torch.distributed.launch` package to run your script like this (replace <num_gpus> with number of gpus):

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/jasper.py ...


Large Training Example
~~~~~~~~~~~~~~~~~~~~~~

Please refer to the `<nemo_git_repo_root>/examples/asr/jasper.py` for comprehensive example. It builds one train DAG and up to three validation DAGs to evaluate on different datasets.

Assuming, you are working with Volta-based DGX, you can run training like this:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/jasper.py --batch_size=64 --num_epochs=100 --lr=0.015 --warmup_steps=8000 --weight_decay=0.001 --train_dataset=/manifests/librivox-train-all.json --eval_datasets /manifests/librivox-dev-clean.json /manifests/librivox-dev-other.json --model_config=<nemo_git_repo_root>/nemo/examples/asr/configs/jasper15x5SEP.yaml --exp_name=MyLARGE-ASR-EXPERIMENT

The command above should trigger 8-GPU training with mixed precision. In the command above various manifests (.json) files are various datasets. Substitute them with the ones containing your data.

.. tip::
    You can pass several manifests (comma-separated) to train on a combined dataset like this: `--train_manifest=/manifests/librivox-train-all.json,/manifests/librivox-train-all-sp10pcnt.json,/manifests/cv/validated.json`. Here it combines 3 data sets: LibriSpeech, Mozilla Common Voice and LibriSpeech speed perturbed.


Fine-tuning
-----------
Training time can be dramatically reduced if starting from a good pre-trained model:

    (1) Obtain pre-trained model (jasper_encoder, jasper_decoder and configuration files) `from here <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`_.
    (2) load pre-trained weights right after you've instantiated your jasper_encoder and jasper_decoder, like this:

.. code-block:: python

    jasper_encoder.restore_from("<path_to_checkpoints>/15x5SEP/JasperEncoder-STEP-247400.pt")
    jasper_decoder.restore_from("<path_to_checkpoints>/15x5SEP/JasperDecoderForCTC-STEP-247400.pt")
    # in case of distributed training add args.local_rank
    jasper_decoder.restore_from("<path_to_checkpoints>/15x5SEP/JasperDecoderForCTC-STEP-247400.pt", args.local_rank)

.. tip::
    When fine-tuning, use smaller learning rate.


Inference
---------

First download pre-trained model (jasper_encoder, jasper_decoder and configuration files) `from here <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`_ into `<path_to_checkpoints>`. We will use this pre-trained model to measure WER on LibriSpeech dev-clean dataset.

.. code-block:: bash

    python <nemo_git_repo_root>/examples/asr/jasper_infer.py --model_config=<nemo_git_repo_root>/examples/asr/configs/jasper15x5SEP.yaml --eval_datasets "<path_to_data>/dev_clean.json" --load_dir=<directory_containing_checkpoints>


Inference with Language Model
-----------------------------

Using KenLM
~~~~~~~~~~~
We will be using `Baidu's CTC decoder with LM implementation. <https://github.com/PaddlePaddle/DeepSpeech>`_.

Perform the following steps:

    * Go to ``cd <nemo_git_repo_root>/scripts``
    * Install Baidu's CTC decoders (NOTE: no need for "sudo" if inside the container):
        * ``sudo apt-get update && sudo apt-get install swig``
        * ``sudo apt-get install pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev``
        * ``sudo apt-get install libsndfile1-dev python-setuptools libboost-all-dev python-dev``
        * ``./install_decoders.sh``
    * Build 6-gram KenLM model on LibriSpeech ``./build_6-gram_OpenSLR_lm.sh``
    * Run jasper_infer.py with the --lm_path flag

    .. code-block:: bash

        python <nemo_git_repo_root>/examples/asr/jasper_infer.py --model_config=<nemo_git_repo_root>/examples/asr/configs/jasper15x5SEP.yaml --eval_datasets "<path_to_data>/dev_clean.json" --load_dir=<directory_containing_checkpoints> --lm_path=<path_to_6gram.binary>


References
----------

.. bibliography:: Jasperbib.bib
    :style: plain
