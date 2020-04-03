Tutorial
========

Make sure you have installed ``nemo`` and the ``nemo_asr`` collection.
See the :ref:`installation` section.

.. note::
  You need to have ``nemo`` and the ``nemo_asr`` collection for this tutorial.
  It is also necessary to install `torchaudio` in order to use MFCC preprocessing.


Introduction
------------

Speech Command Recognition is the task of classifying an input audio pattern into a discrete set of classes.
It is a subset of Automatic Speech Recognition, sometimes referred to as Key Word Spotting, in which a model is constantly analyzing speech patterns to detect certain "command" classes.
Upon detection of these commands, a specific action can be taken by the system. It is often the objective of command recognition models to be small and efficient, so that they can be deployed onto
low power sensors and remain active for long durations of time.

This Speech Command recognition tutorial is based on the QuartzNet model :cite:`speech-recognition-tut-kriman2019quartznet` with
a modified decoder head to suit classification tasks. Instead of predicting a token for each time step of the input, we predict
a single label for the entire duration of the audio signal. This is accomplished by a decoder head that performs Global Max / Average pooling
across all timesteps prior to classification. After this, the model can be trained via standard categorical cross-entropy loss.

1. Audio preprocessing (feature extraction): signal normalization, windowing, (log) spectrogram (or mel scale spectrogram, or MFCC)
2. Data augmentation using SpecAugment :cite:`speech-recognition-tut-park2019` to increase number of data samples.
3. Develop a small Neural classification model which can be trained efficiently.

.. note::
  A Jupyter Notebook containing all the steps to download the dataset, train a model and evaluate its results
  is available at : `Speech Commands Using NeMo <https://github.com/NVIDIA/NeMo/blob/master/examples/asr/notebooks/3_Speech_Commands_using_NeMo.ipynb>`_

Data Preparation
----------------

We will be using the open source Google Speech Commands Dataset (we will use V1 of the dataset for the tutorial, but require
very minor changes to support V2 dataset). These scripts below will download the dataset and convert it to a format suitable
for use with `nemo_asr`:


.. code-block:: bash

    mkdir data
    # process_speech_commands_data.py script is located under <nemo_git_repo_root>/scripts
    # The `--rebalance` flag will duplicate elements in the train set so that all classes
    # have the same number of elements. It is not mandatory to add this flag.
    python process_speech_commands_data.py --data_root=data --data_version=1 --rebalance

.. note::
    You should have at least 4GB of disk space available if you've used ``--data_version=1``; and at least 6GB if you used ``--data_version=2``. Also, it will take some time to download and process, so go grab a coffee.

After download and conversion, your `data` folder should contain a directory called `google_speech_recognition_v{1/2}`.
Inside this directory, there should be multiple subdirectory containing wav files, and three json manifest files:

* `train_manifest.json`
* `validation_manifest.json`
* `test_manifest.json`

Each line in json file describes a training sample - `audio_filepath` contains path to the wav file, `duration` it's duration in seconds, and `label` is the class label:

.. code-block:: json

    {"audio_filepath": "<absolute path to dataset>/two/8aa35b0c_nohash_0.wav", "duration": 1.0, "command": "two"}
    {"audio_filepath": "<absolute path to dataset>/two/ec5ab5d5_nohash_2.wav", "duration": 1.0, "command": "two"}


Training
---------

We will be training a QuartzNet model :cite:`speech-recognition-tut-kriman2019quartznet`.
The benefit of QuartzNet over JASPER models is that they use Separable Convolutions, which greatly reduce the number of
parameters required to get good model accuracy.

QuartzNet models generally follow the model definition pattern QuartzNet-[BxR], where B is the number of blocks and R is the number of
convolutional sub-blocks. Each sub-block contains a 1-D masked convolution, batch normalization, ReLU, and dropout:

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model

In the tutorial we will be using model QuartzNet [3x1].
The script below does both training and evaluation (on V1 dataset) on single GPU:

    .. tip::
        Run Jupyter notebook and walk through this script step-by-step


**Training script**

.. code-block:: python

    # Import some utility functions
    import argparse
    import copy
    import math
    import os
    import glob
    from functools import partial
    from datetime import datetime
    from ruamel.yaml import YAML

    # NeMo's "core" package
    import nemo
    # NeMo's ASR collection
    import nemo.collections.asr as nemo_asr
    # NeMo's learning rate policy
    from nemo.utils.lr_policies import CosineAnnealing
    from nemo.collections.asr.helpers import (
        monitor_classification_training_progress,
        process_classification_evaluation_batch,
        process_classification_evaluation_epoch,
    )

    logging = nemo.logging

    # Lets define some hyper parameters
    lr = 0.05
    num_epochs = 100
    batch_size = 128
    weight_decay = 0.001

    # Create a Neural Factory
    # It creates log files and tensorboard writers for us among other functions
    neural_factory = nemo.core.NeuralModuleFactory(
        log_dir='./quartznet-3x1-v1',
        create_tb_writer=True)
    tb_writer = neural_factory.tb_writer

    # Path to our training manifest
    train_dataset = "<path_to_where_you_put_data>/train_manifest.json"

    # Path to our validation manifest
    eval_datasets = "<path_to_where_you_put_data>/test_manifest.json"

    # Here we will be using separable convolutions
    # with 3 blocks (k=3 repeated once r=1 from the picture above)
    yaml = YAML(typ="safe")
    with open("<nemo_git_repo_root>/examples/asr/configs/quartznet_speech_commands_3x1_v1.yaml") as f:
        jasper_params = yaml.load(f)

    # Pre-define a set of labels that this model must learn to predict
    labels = jasper_params['labels']

    # Get the sampling rate of the data
    sample_rate = jasper_params['sample_rate']

    # Check if data augmentation such as white noise and time shift augmentation should be used
    audio_augmentor = jasper_params.get('AudioAugmentor', None)

    # Build the input data layer and the preprocessing layers for the train set
    train_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
        manifest_filepath=train_dataset,
        labels=labels,
        sample_rate=sample_rate,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        augmentor=audio_augmentor,
        shuffle=True
    )

     # Build the input data layer and the preprocessing layers for the test set
    eval_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
        manifest_filepath=eval_datasets,
        sample_rate=sample_rate,
        labels=labels,
        batch_size=args.eval_batch_size,
        num_workers=os.cpu_count(),
        shuffle=False,
    )

    # We will convert the raw audio data into MFCC Features to feed as input to our model
    data_preprocessor = nemo_asr.AudioToMFCCPreprocessor(
        sample_rate=sample_rate, **jasper_params["AudioToMFCCPreprocessor"],
    )

    # Compute the total number of samples and the number of training steps per epoch
    N = len(train_data_layer)
    steps_per_epoch = math.ceil(N / float(args.batch_size))

    logging.info("Steps per epoch : {0}".format(steps_per_epoch))
    logging.info('Have {0} examples to train on.'.format(N))

    # Here we begin defining all of the augmentations we want
    # We will pad the preprocessed spectrogram image to have a certain number of timesteps
    # This centers the generated spectrogram and adds black boundaries to either side
    # of the padded image.
    crop_pad_augmentation = nemo_asr.CropOrPadSpectrogramAugmentation(audio_length=128)

    # We also optionally add `SpecAugment` augmentations based on the config file
    # SpecAugment has various possible augmentations to the generated spectrogram
    # 1) Frequency band masking
    # 2) Time band masking
    # 3) Rectangular cutout
    spectr_augment_config = jasper_params.get('SpectrogramAugmentation', None)
    if spectr_augment_config:
        data_spectr_augmentation = nemo_asr.SpectrogramAugmentation(**spectr_augment_config)

    # Build the QuartzNet Encoder model
    # The config defines the layers as a list of dictionaries
    # The first and last two blocks are not considered when we say QuartzNet-[BxR]
    # B is counted as the number of blocks after the first layer and before the penultimate layer.
    # R is defined as the number of repetitions of each block in B.
    # Note: We can scale the convolution kernels size by the float parameter `kernel_size_factor`
    jasper_encoder = nemo_asr.JasperEncoder(**jasper_params["JasperEncoder"])

    # We then define the QuartzNet decoder.
    # This decoder head is specialized for the task for classification, such that it
    # accepts a set of `N-feat` per timestep of the model, and averages these features
    # over all the timesteps, before passing a Linear classification layer on those features.
    jasper_decoder = nemo_asr.JasperDecoderForClassification(
        feat_in=jasper_params["JasperEncoder"]["jasper"][-1]["filters"],
        num_classes=len(labels),
        **jasper_params['JasperDecoderForClassification'],
    )

    # We can easily apply cross entropy loss to train this model
    ce_loss = nemo_asr.CrossEntropyLossNM()

    # Lets print out the number of parameters of this model
    logging.info('================================')
    logging.info(f"Number of parameters in encoder: {jasper_encoder.num_weights}")
    logging.info(f"Number of parameters in decoder: {jasper_decoder.num_weights}")
    logging.info(
        f"Total number of parameters in model: " f"{jasper_decoder.num_weights + jasper_encoder.num_weights}"
    )
    logging.info('================================')

    # Now we have all of the components that are required to build the NeMo execution graph!
    ## Build the training data loaders and preprocessors first
    audio_signal, audio_signal_len, commands, command_len = train_data_layer()
    processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)
    processed_signal, processed_signal_len = crop_pad_augmentation(
        input_signal=processed_signal,
        length=audio_signal_len
    )

    ## Augment the dataset for training
    if spectr_augment_config:
        processed_signal = data_spectr_augmentation(input_spec=processed_signal)

    ## Define the model
    encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
    decoded = jasper_decoder(encoder_output=encoded)

    ## Obtain the train loss
    train_loss = ce_loss(logits=decoded, labels=commands)

    # Now we build the test graph in a similar way, reusing the above components
    ## Build the test data loader and preprocess same way as train graph
    ## But note, we do not add the spectrogram augmentation to the test graph !
    test_audio_signal, test_audio_signal_len, test_commands, test_command_len = eval_data_layer()
    test_processed_signal, test_processed_signal_len = data_preprocessor(
        input_signal=test_audio_signal, length=test_audio_signal_len
    )
    test_processed_signal, test_processed_signal_len = crop_pad_augmentation(
        input_signal=test_processed_signal, length=test_processed_signal_len
    )

    # Pass the test data through the model encoder and decoder
    test_encoded, test_encoded_len = jasper_encoder(
        audio_signal=test_processed_signal, length=test_processed_signal_len
    )
    test_decoded = jasper_decoder(encoder_output=test_encoded)

    # Compute test loss for visualization
    test_loss = ce_loss(logits=test_decoded, labels=test_commands)

    # Now that we have our training and evaluation graphs built,
    # we can focus on a few callbacks to help us save the model checkpoints
    # during training, as well as display train and test metrics

    # Callbacks needed to print train info to console and Tensorboard
    train_callback = nemo.core.SimpleLossLoggerCallback(
        # Notice that we pass in loss, predictions, and the labels.
        # Of course we would like to see our training loss, but we need the
        # other arguments to calculate the accuracy.
        tensors=[train_loss, decoded, commands],
        # The print_func defines what gets printed.
        print_func=partial(monitor_classification_training_progress, eval_metric=None),
        get_tb_values=lambda x: [("loss", x[0])],
        tb_writer=neural_factory.tb_writer,
    )

    # Callbacks needed to print test info to console and Tensorboard
    tagname = 'TestSet'
    eval_callback = nemo.core.EvaluatorCallback(
        eval_tensors=[test_loss, test_decoded, test_commands],
        user_iter_callback=partial(process_classification_evaluation_batch, top_k=1),
        user_epochs_done_callback=partial(process_classification_evaluation_epoch, eval_metric=1, tag=tagname),
        eval_step=200,  # How often we evaluate the model on the test set
        tb_writer=neural_factory.tb_writer,
    )

    # Callback to save model checkpoints
    chpt_callback = nemo.core.CheckpointCallback(
        folder=neural_factory.checkpoint_dir,
        step_freq=1000,
    )

    # Prepare a list of checkpoints to pass to the engine
    callbacks = [train_callback, eval_callback, chpt_callback]

    # Now we have all the components required to train the model
    # Lets define a learning rate schedule

    # Define a learning rate schedule
    lr_policy = CosineAnnealing(
        total_steps=num_epochs * steps_per_epoch,
        warmup_ratio=0.05,
        min_lr=0.001,
    )

    logging.info(f"Using `{lr_policy}` Learning Rate Scheduler")

    # Finally, lets train this model !
    neural_factory.train(
        tensors_to_optimize=[train_loss],
        callbacks=callbacks,
        lr_policy=lr_policy,
        optimizer="novograd",
        optimization_params={
            "num_epochs": num_epochs,
            "max_steps": None,
            "lr": lr,
            "momentum": 0.95,
            "betas": (0.98, 0.5),
            "weight_decay": weight_decay,
            "grad_norm_clip": None,
        },
        batches_per_step=1,
    )

.. note::
    This script trains should finish 100 epochs in about 4-5 hours on GTX 1080.

.. tip::
    To improve your accuracy:
        (1) Train longer (200-300 epochs)
        (2) Train on more data (try increasing the augmentation parameters for SpectrogramAugmentation)
        (3) Use larger model
        (4) Train on several GPUs and use mixed precision (on NVIDIA Volta and Turing GPUs)
        (5) Start with pre-trained checkpoints


Mixed Precision training
-------------------------
Mixed precision and distributed training in NeMo is based on `NVIDIA's APEX library <https://github.com/NVIDIA/apex>`_.
Make sure it is installed prior to attempting mixed precision training.

To train with mixed-precision all you need is to set `optimization_level` parameter of `nemo.core.NeuralModuleFactory`  to `nemo.core.Optimization.mxprO1`. For example:

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=nemo.core.Optimization.mxprO1,
        placement=nemo.core.DeviceType.AllGpu,
        cudnn_benchmark=True)


Multi-GPU training
-------------------

Enabling multi-GPU training with NeMo is easy:

   (1) First set `placement` to `nemo.core.DeviceType.AllGpu` in NeuralModuleFactory and in your Neural Modules
   (2) Have your script accept 'local_rank' argument and do not set it yourself: `parser.add_argument("--local_rank", default=None, type=int)`
   (3) Use `torch.distributed.launch` package to run your script like this (replace <num_gpus> with number of gpus):

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/quartznet_speech_commands.py ...

.. note::
    Because mixed precision requires Tensor Cores it only works on NVIDIA Volta and Turing based GPUs

Large Training Example
~~~~~~~~~~~~~~~~~~~~~~

Please refer to the `<nemo_git_repo_root>/examples/asr/quartznet_speech_commands.py` for comprehensive example.
It builds one train DAG, one validation DAG and a test DAG to evaluate on different datasets.

Assuming, you are working with Volta-based DGX, you can run training like this:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/quartznet_speech_commands.py --model_config "<nemo_git_repo_root>/examples/asr/configs/quartznet_speech_commands_3x1_v1.yaml" \
      --train_dataset="<absolute path to dataset>/train_manifest.json" --eval_datasets "<absolute path to dataset>/validation_manifest.json" "<absolute path to dataset>/test_manifest.json" \
      --num_epochs=200 --batch_size=128 --eval_batch_size=128 --eval_freq=200 --lr=0.05 --min_lr=0.001 \
      --optimizer="novograd" --weight_decay=0.001 --amp_opt_level="O1" --warmup_ratio=0.05 --hold_ratio=0.45 \
      --checkpoint_dir="./checkpoints/quartznet_speech_commands_checkpoints_3x1_v1/" \
      --exp_name="./results/quartznet_speech_classification-quartznet-3x1_v1/"

The command above should trigger 8-GPU training with mixed precision. In the command above various manifests (.json) files are various datasets. Substitute them with the ones containing your data.

.. tip::
    You can pass several manifests (comma-separated) to train on a combined dataset like this: `--train_manifest=/manifests/<first dataset>.json,/manifests/<second dataset>.json`


Fine-tuning
-----------
Training time can be dramatically reduced if starting from a good pre-trained model:

    (1) Obtain pre-trained model (jasper_encoder, jasper_decoder and configuration files).
    (2) load pre-trained weights right after you've instantiated your jasper_encoder and jasper_decoder, like this:

.. code-block:: python

    jasper_encoder.restore_from("<path_to_checkpoints>/JasperEncoder-STEP-89000.pt")
    jasper_decoder.restore_from("<path_to_checkpoints>/JasperDecoderForClassification-STEP-89000.pt")
    # in case of distributed training add args.local_rank
    jasper_decoder.restore_from("<path_to_checkpoints>/JasperDecoderForClassification-STEP-89000.pt", args.local_rank)

.. tip::
    When fine-tuning, use smaller learning rate.


Evaluation
----------

First download pre-trained model (jasper_encoder, jasper_decoder and configuration files) into `<path_to_checkpoints>`.
We will use this pre-trained model to measure classification accuracy on Google Speech Commands dataset v1,
but they can similarly be used for v2 dataset.

.. note::
    To listen to the samples that were incorrectly labeled by the model, please run the following code in a notebook.

.. code-block:: python

    # Lets add some generic imports.
    # Please note that you will need to install `librosa` for this code
    # To install librosa : Run `!pip install librosa` from the notebook itself.
    import glob
    import os
    import json
    import re
    import numpy as np
    import torch
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import IPython.display as ipd
    from ruamel.yaml import YAML

    # Import nemo and asr collections
    import nemo
    import nemo.collections.asr as nemo_asr

    logging = nemo.logging

    # We add some
    data_dir = '<path to the data directory>'
    data_version = 1
    config_path = '<path to the config file for this model>'
    model_path = '<path to the checkpoint directory for this model>'

    test_manifest = os.path.join(data_dir, "test_manifest.json")

    # Parse the config file provided to us
    # Parse config and pass to model building function
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
        logging.info("******\nLoaded config file.\n******")

    labels = params['labels']  # Vocab of tokens
    sample_rate = params['sample_rate']
    batch_size = 128

    # Build the evaluation graph
    # Create our NeuralModuleFactory, which will oversee the neural modules.
    neural_factory = nemo.core.NeuralModuleFactory(
        log_dir=f'v{data_version}/eval_results/')

    logger = neural_factory.logger

    test_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
        manifest_filepath=test_manifest,
        labels=labels,
        sample_rate=sample_rate,
        shuffle=False,
        batch_size=batch_size,
    )
    crop_pad_augmentation = nemo_asr.CropOrPadSpectrogramAugmentation(
        audio_length=128
    )
    data_preprocessor = nemo_asr.AudioToMFCCPreprocessor(
        sample_rate=sample_rate,
        **params['AudioToMFCCPreprocessor']
    )

    # Create the Jasper_3x1 encoder as specified, and a classification decoder
    encoder = nemo_asr.JasperEncoder(**params['JasperEncoder'])
    decoder = nemo_asr.JasperDecoderForClassification(
        feat_in=params['JasperEncoder']['jasper'][-1]['filters'],
        num_classes=len(labels),
        **params['JasperDecoderForClassification']
    )

    ce_loss = nemo_asr.CrossEntropyLossNM()

    # Assemble the DAG components
    test_audio_signal, test_audio_signal_len, test_commands, test_command_len = test_data_layer()

    test_processed_signal, test_processed_signal_len = data_preprocessor(
        input_signal=test_audio_signal,
        length=test_audio_signal_len
    )

    # --- Crop And Pad Augment --- #
    test_processed_signal, test_processed_signal_len = crop_pad_augmentation(
        input_signal=test_processed_signal,
        length=test_processed_signal_len
    )

    test_encoded, test_encoded_len = encoder(
        audio_signal=test_processed_signal,
        length=test_processed_signal_len
    )

    test_decoded = decoder(
        encoder_output=test_encoded
    )

    test_loss = ce_loss(
        logits=test_decoded,
        labels=test_commands
    )

    # We import the classification accuracy metric to compute Top-1 accuracy
    from nemo.collections.asr.metrics import classification_accuracy
    from functools import partial

    # --- Inference Only --- #
    # We've already built the inference DAG above, so all we need is to call infer().
    evaluated_tensors = neural_factory.infer(
        # These are the tensors we want to get from the model.
        tensors=[test_loss, test_decoded, test_commands],
        # checkpoint_dir specifies where the model params are loaded from.
        checkpoint_dir=model_path
        )

    # Let us count the total number of incorrect classifications by this model
    correct_count = 0
    total_count = 0

    for batch_idx, (logits, labels) in enumerate(zip(evaluated_tensors[1], evaluated_tensors[2])):
        acc = classification_accuracy(
            logits=logits,
            targets=labels,
            top_k=[1]
        )

        # Select top 1 accuracy only
        acc = acc[0]

        # Since accuracy here is "per batch", we simply denormalize it by multiplying
        # by batch size to recover the count of correct samples.
        correct_count += int(acc * logits.size(0))
        total_count += logits.size(0)

    logging.info(f"Total correct / Total count : {correct_count} / {total_count}")
    logging.info(f"Final accuracy : {correct_count / float(total_count)}")

    # Let us now filter out the incorrectly labeled samples from the total set of samples in the test set

    # First lets create a utility class to remap the integer class labels to actual string label
    class ReverseMapLabel:
        def __init__(self, data_layer: nemo_asr.AudioToSpeechLabelDataLayer):
            self.label2id = dict(data_layer._dataset.label2id)
            self.id2label = dict(data_layer._dataset.id2label)

        def __call__(self, pred_idx, label_idx):
            return self.id2label[pred_idx], self.id2label[label_idx]

    # Next, lets get the indices of all the incorrectly labeled samples
    sample_idx = 0
    incorrect_preds = []
    rev_map = ReverseMapLabel(test_data_layer)

    for batch_idx, (logits, labels) in enumerate(zip(evaluated_tensors[1], evaluated_tensors[2])):
        probs = torch.softmax(logits, dim=-1)
        probas, preds = torch.max(probs, dim=-1)

        incorrect_ids = (preds != labels).nonzero()
        for idx in incorrect_ids:
            proba = float(probas[idx][0])
            pred = int(preds[idx][0])
            label = int(labels[idx][0])
            idx = int(idx[0]) + sample_idx

            incorrect_preds.append((idx, *rev_map(pred, label), proba))

        sample_idx += labels.size(0)

    logging.info(f"Num test samples : {total_count}")
    logging.info(f"Num errors : {len(incorrect_preds)}")

    # First lets sort by confidence of prediction
    incorrect_preds = sorted(incorrect_preds, key=lambda x: x[-1], reverse=False)

    # Lets print out the (test id, predicted label, ground truth label, confidence)
    # tuple of first 20 incorrectly labeled samples
    for incorrect_sample in incorrect_preds[:20]:
        logging.info(str(incorrect_sample))

    # Lets define a threshold below which we designate a model's prediction as "low confidence"
    # and then filter out how many such samples exist
    low_confidence_threshold = 0.25
    count_low_confidence = len(list(filter(lambda x: x[-1] <= low_confidence_threshold, incorrect_preds)))
    logging.info(f"Number of low confidence predictions : {count_low_confidence}")

    # One interesting observation is to actually listen to these samples whose predicted labels were incorrect
    # Note: The following requires the use of a Notebook environment

    # First lets create a helper function to parse the manifest files
    def parse_manifest(manifest):
        data = []
        for line in manifest:
            line = json.loads(line)
            data.append(line)

        return data

    # Now lets load the test manifest into memory
    test_samples = []
    with open(test_manifest, 'r') as test_f:
        test_samples = test_f.readlines()

    test_samples = parse_manifest(test_samples)

    # Next, lets create a helper function to actually listen to certain samples
    def listen_to_file(sample_id, pred=None, label=None, proba=None):
        # Load the audio waveform using librosa
        filepath = test_samples[sample_id]['audio_filepath']
        audio, sample_rate = librosa.load(filepath)

        if pred is not None and label is not None and proba is not None:
            logging.info(f"Sample : {sample_id} Prediction : {pred} Label : {label} Confidence = {proba: 0.4f}")
        else:
            logging.info(f"Sample : {sample_id}")

        return ipd.Audio(audio, rate=sample_rate)

    # Finally, lets listen to all the audio samples where the model made a mistake
    # Note: This list of incorrect samples may be quite large, so you may choose to subsample `incorrect_preds`
    for sample_id, pred, label, proba in incorrect_preds:
        ipd.display(listen_to_file(sample_id, pred=pred, label=label, proba=proba))  # Needs to be run in a notebook environment

References
----------

.. bibliography:: speech_recognition_all.bib
    :style: plain
    :labelprefix: SPEECH-RECOGNITION-ALL-TUT
    :keyprefix: speech-recognition-tut-
