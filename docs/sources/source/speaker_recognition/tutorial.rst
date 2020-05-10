Tutorial
========

Make sure you have installed ``nemo`` and the ``nemo_asr`` collection.
See the :ref:`installation` section.

.. note::
    
    You need to have ``nemo`` and the ``nemo_asr`` collection for this tutorial.
    It is also necessary to install `torchaudio` in order to use MFCC preprocessing.


Introduction
------------

Speaker Recognition (SR) is an broad research area which solves two major tasks: speaker identification (who is speaking?) and 
speaker verification (is the speaker who she claims to be?). In this work, we focus on the far-field, 
text-independent speaker recognition when the identity of the speaker is based on how speech is spoken, 
not necessarily in what is being said. Typically such SR systems operate on unconstrained speech utterances, 
which are converted into vector of fixed length, called speaker embedding. Speaker embedding is also  used in 
automatic speech recognition (ASR) and speech synthesis. 

As goal of most speaker related systems is to get good speaker level embeddings that could help distinguish from other speakers, we shall first train these embeddings in end-to-end
manner optimizing the QuatzNet based :cite:`speaker-tut-kriman2019quartznet` encoder model on cross-entropy loss. 
We modeify the decoder to get these fixed size embeddings irrespective of length of input audio. We employ mean and variance 
based statistics pooling method to grab these embeddings.

In this tutorial we shall first train these embeddings on speaker related datasets and then get speaker embeddings from a 
pretrained network for a new dataset, then followed by scoring them using cosine similarity method or optionally with PLDA backend. 

.. .. note::
..   A Jupyter Notebook containing all the steps to download the dataset, train a model and evaluate its results
..   is available at : `Speech Commands Using NeMo <https://github.com/NVIDIA/NeMo/blob/master/examples/speaker_recognition/notebooks/3_Speech_Commands_using_NeMo.ipynb>`_

Data Preparation
----------------

Before proceeding to next steps please make sure you downloaded required datasets. For this tutorial I'll be using hi-mia. But 
for better results use most of the data available from voxceleb1 :cite:`speaker-tut-nagrani2017voxceleb`, voxceleb2 :cite:`speaker-tut-nagrani2017voxceleb`, HI-MIA :cite:`speaker-tut-himia`, AISHELL which are 
most common speaker related datasets. Once you have these or related datasets, we need to generate manifest files that would be 
used to train the network with `nemo_asr`. Steps and scripts below will help you generate one for yours. These scripts are 
present in <NeMo_root>/scripts

.. code-block:: bash 
    
    cd <NeMo_root>/scripts 
    mkdir data 
    python get_hi-mia_data.py --data_root=data 

This script downloads data, extracts files, converts audio to 16Khz and creates manifest files for train, dev and test inside respective directories with 
format {set}_all.json. It also creates train and dev manifest files in each train and dev directories that are split on 
stratified basis on speakers. So your data folder looks like :

After download and conversion, your `data` folder should contain a directories with manifest files as:

* `data/<set>/train.json`
* `data/<set>/dev.json` 
* `data/<set>/{set}_all.json` 

Also for each set we also create utt2spk files, these files later would be used in PLDA training.

Each line in manifest file describes a training sample - `audio_filepath` contains path to the wav file, `duration` it's duration in seconds, and `label` is the speaker class label:

.. code-block:: json

    {"audio_filepath": "<absolute path to dataset>/data/train/SPEECHDATA/wav/SV0184/SV0184_6_04_N3430.wav", "duration": 1.22, "label": "SV0184"}
    {"audio_filepath": "<absolute path to dataset>/data/train/SPEECHDATA/wav/SV0184/SV0184_5_03_F2037.wav", "duration": 1.375, "label": "SV0184"}


If you would like to generate the same set of files for other datasets, you may follow below steps for those datasets.

.. code-block:: bash

    mkdir -p data_voxceleb #or relevant datasetname 
    cd <set> # repeat for train, dev and test 
    find $PWD/{dataset with wav files} -iname *.wav > all_wav.scp 
    # once we have scp files for all datasets for each dataset run the below script to get respective dev 
    # train manifest files.
    # all scripts are located under <nemo_root>/scripts
    # scp_to_manifest.py will take arguments as scp file, id from filename separated by  '/' 
    # to be considered as speaker label and out_put manifest file name 
    # if filename is /data/SSD/files/DATASETS/voxceleb/data/dev/aac_wav/id01192/Q0k4WGaT8ZM/00084.wav then id is 9 which 
    # corresponds to speaker label id01192
    # pass --split option to split <set> manifest file based on stratified split on speaker basis (10%) you may not need this for test set
    python scp_to_manifest.py --scp='all_wav.scp' --id=9 --out='all_manifest.json' --split
    # this will create two files <manifest_out>.json and <manifest_out>.json in current directory 
    # repeat for all your datasets

Training
---------

We will be training a QuartzNet model :cite:`speaker-tut-kriman2019quartznet`. The benefit of QuartzNet over JASPER models is that they use Separable Convolutions, 
which greatly reduce the number of parameters required to get good model accuracy.
    
QuartzNet models generally follow the model definition pattern QuartzNet-[BxR], where B is the number of blocks and R is the number of
convolutional sub-blocks. Each sub-block contains a 1-D masked convolution, batch normalization, ReLU, and dropout:

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model

In the tutorial we will be using model QuartzNet [3x2]. with narrow filters, whole config can be found in `examples/speaker_recognition/configs/`
The script below which is in  <nemo/examples/speaker_recognition/speaker_reco.py> with below command does both training and evaluation on train set on single GPU:

.. code-block:: bash

    python speaker_reco.py --batch_size=128 --optimizer='novograd' 
    --num_epochs=25 --model_config="<./configs/quartznet_spkr_3x2x512_xvector.yaml" --emb_size=1024 \
    --eval_datasets '<data_root>/train/dev.json' \
    --train_dataset='<data_root>/train/train.json' \
    --checkpoint_dir='./myExps/checkpoints/' --print_freq=400 --synced_bn \
    --checkpoint_save_freq=1000 --create_tb_writer  --eval_freq=1000  \
    --exp_name='quartznet3x2x512_himia'  --iter_per_step=1  \
    --lr=0.02  --lr_policy='CosineAnnealing' --eval_batch_size=64 \
    --tensorboard_dir='./myExps/tensorboard/'  --warmup_steps=1000  \
    --weight_decay=0.001 --work_dir='./myExps/'

.. .. tip::
..     Run Jupyter notebook and walk through this script step-by-step


**Training script**

.. code-block:: python

    import argparse
    import copy
    import os
    from functools import partial

    from ruamel.yaml import YAML

    import nemo
    import nemo.collections.asr as nemo_asr
    import nemo.utils.argparse as nm_argparse
    from nemo.collections.asr.helpers import (
        monitor_classification_training_progress,
        process_classification_evaluation_batch,
        process_classification_evaluation_epoch,
    )
    from nemo.utils.lr_policies import CosineAnnealing

    logging = nemo.logging


    def parse_args():
        parser = argparse.ArgumentParser(
            parents=[nm_argparse.NemoArgParser()], description="SpeakerRecognition", conflict_handler="resolve",
        )
        parser.set_defaults(
            checkpoint_dir=None,
            optimizer="novograd",
            batch_size=32,
            eval_batch_size=64,
            lr=0.01,
            weight_decay=0.001,
            amp_opt_level="O1",
            create_tb_writer=True,
        )

        # Overwrite default args
        parser.add_argument(
            "--num_epochs",
            type=int,
            default=None,
            required=True,
            help="number of epochs to train. You should specify either num_epochs or max_steps",
        )
        parser.add_argument(
            "--model_config", type=str, required=True, help="model configuration file: model.yaml",
        )

        # Create new args
        parser.add_argument("--exp_name", default="SpkrReco_GramMatrix", type=str)
        parser.add_argument("--beta1", default=0.95, type=float)
        parser.add_argument("--beta2", default=0.5, type=float)
        parser.add_argument("--warmup_steps", default=1000, type=int)
        parser.add_argument("--load_dir", default=None, type=str)
        parser.add_argument("--synced_bn", action="store_true", help="Use synchronized batch norm")
        parser.add_argument("--emb_size", default=256, type=int)
        parser.add_argument("--synced_bn_groupsize", default=0, type=int)
        parser.add_argument("--print_freq", default=256, type=int)

        args = parser.parse_args()
        if args.max_steps is not None:
            raise ValueError("QuartzNet uses num_epochs instead of max_steps")

        return args


    def construct_name(name, lr, batch_size, num_epochs, wd, optimizer, emb_size):
        return "{0}-lr_{1}-bs_{2}-e_{3}-wd_{4}-opt_{5}-embsize_{6}".format(
            name, lr, batch_size, num_epochs, wd, optimizer, emb_size
        )


    def create_all_dags(args, neural_factory):
        """
        creates train and eval dags as well as their callbacks
        returns train loss tensor and callbacks"""

        # parse the config files
        yaml = YAML(typ="safe")
        with open(args.model_config) as f:
            spkr_params = yaml.load(f)

        sample_rate = spkr_params["sample_rate"]
        time_length = spkr_params.get("time_length", 8)
        logging.info("max time length considered is {} sec".format(time_length))

        # Calculate num_workers for dataloader
        total_cpus = os.cpu_count()
        cpu_per_traindl = max(int(total_cpus / neural_factory.world_size), 1) // 2

        # create data layer for training
        train_dl_params = copy.deepcopy(spkr_params["AudioToSpeechLabelDataLayer"])
        train_dl_params.update(spkr_params["AudioToSpeechLabelDataLayer"]["train"])
        del train_dl_params["train"]
        del train_dl_params["eval"]
        audio_augmentor = spkr_params.get("AudioAugmentor", None)
        # del train_dl_params["normalize_transcripts"]

        data_layer_train = nemo_asr.AudioToSpeechLabelDataLayer(
            manifest_filepath=args.train_dataset,
            labels=None,
            batch_size=args.batch_size,
            num_workers=cpu_per_traindl,
            augmentor=audio_augmentor,
            time_length=time_length,
            **train_dl_params,
            # normalize_transcripts=False
        )

        N = len(data_layer_train)
        steps_per_epoch = int(N / (args.batch_size * args.iter_per_step * args.num_gpus))

        logging.info("Number of steps per epoch {}".format(steps_per_epoch))
        # create separate data layers for eval
        # we need separate eval dags for separate eval datasets
        # but all other modules in these dags will be shared

        eval_dl_params = copy.deepcopy(spkr_params["AudioToSpeechLabelDataLayer"])
        eval_dl_params.update(spkr_params["AudioToSpeechLabelDataLayer"]["eval"])
        del eval_dl_params["train"]
        del eval_dl_params["eval"]

        data_layers_test = []
        for test_set in args.eval_datasets:

            data_layer_test = nemo_asr.AudioToSpeechLabelDataLayer(
                manifest_filepath=test_set,
                labels=data_layer_train.labels,
                batch_size=args.batch_size,
                num_workers=cpu_per_traindl,
                time_length=time_length,
                **eval_dl_params,
                # normalize_transcripts=False
            )
            data_layers_test.append(data_layer_test)
        # create shared modules

        data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate, **spkr_params["AudioToMelSpectrogramPreprocessor"],
        )

        spectr_augment_config = spkr_params.get("SpectrogramAugmentation", None)
        if spectr_augment_config:
            data_spectr_augmentation = nemo_asr.SpectrogramAugmentation(**spectr_augment_config)
        # (QuartzNet uses the Jasper baseline encoder and decoder)
        encoder = nemo_asr.JasperEncoder(**spkr_params["JasperEncoder"],)

        decoder = nemo_asr.JasperDecoderForSpkrClass(
            feat_in=spkr_params["JasperEncoder"]["jasper"][-1]["filters"],
            num_classes=data_layer_train.num_classes,
            pool_mode=spkr_params["JasperDecoderForSpkrClass"]['pool_mode'],
            emb_sizes=spkr_params["JasperDecoderForSpkrClass"]["emb_sizes"].split(","),
        )
        if os.path.exists(args.checkpoint_dir + "/JasperEncoder-STEP-100.pt"):
            encoder.restore_from(args.checkpoint_dir + "/JasperEncoder-STEP-100.pt")
            logging.info("Pretrained Encoder loaded")

        weight = None
        xent_loss = nemo_asr.CrossEntropyLossNM(weight=weight)

        # assemble train DAG

        audio_signal, audio_signal_len, label, label_len = data_layer_train()

        processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)

        if spectr_augment_config:
            processed_signal = data_spectr_augmentation(input_spec=processed_signal)

        encoded, encoded_len = encoder(audio_signal=processed_signal, length=processed_signal_len)

        logits, _ = decoder(encoder_output=encoded)
        loss = xent_loss(logits=logits, labels=label)

        # create train callbacks
        train_callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss, logits, label],
            print_func=partial(monitor_classification_training_progress, eval_metric=[1]),
            step_freq=args.print_freq,
            get_tb_values=lambda x: [("train_loss", x[0])],
            tb_writer=neural_factory.tb_writer,
        )

        callbacks = [train_callback]

        if args.checkpoint_dir or args.load_dir:
            chpt_callback = nemo.core.CheckpointCallback(
                folder=args.checkpoint_dir,
                load_from_folder=args.checkpoint_dir,  # load dir
                step_freq=args.checkpoint_save_freq,
                checkpoints_to_keep=125,
            )

            callbacks.append(chpt_callback)

        # --- Assemble Validation DAG --- #

        for i, eval_layer in enumerate(data_layers_test):

            audio_signal_test, audio_len_test, label_test, _ = eval_layer()
            processed_signal_test, processed_len_test = data_preprocessor(
                input_signal=audio_signal_test, length=audio_len_test
            )
            encoded_test, encoded_len_test = encoder(audio_signal=processed_signal_test, length=processed_len_test)
            logits_test, _ = decoder(encoder_output=encoded_test)
            loss_test = xent_loss(logits=logits_test, labels=label_test)

            tagname = os.path.dirname(args.eval_datasets[i]).split("/")[-1] + "_" + str(i)
            print(tagname)
            eval_callback = nemo.core.EvaluatorCallback(
                eval_tensors=[loss_test, logits_test, label_test],
                user_iter_callback=partial(process_classification_evaluation_batch, top_k=1),
                user_epochs_done_callback=partial(process_classification_evaluation_epoch, tag=tagname),
                eval_step=args.eval_freq,  # How often we evaluate the model on the test set
                tb_writer=neural_factory.tb_writer,
            )

            callbacks.append(eval_callback)

        return loss, callbacks, steps_per_epoch, loss_test, logits_test, label_test


    def main():
        args = parse_args()

        print(args)
        emb_size = 1024
        name = construct_name(
            args.exp_name, args.lr, args.batch_size, args.num_epochs, args.weight_decay, args.optimizer, emb_size=emb_size,
        )
        work_dir = name
        if args.work_dir:
            work_dir = os.path.join(args.work_dir, name)

        # instantiate Neural Factory with supported backend
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch,
            local_rank=args.local_rank,
            optimization_level=args.amp_opt_level,
            log_dir=work_dir,
            checkpoint_dir=args.checkpoint_dir + "/" + args.exp_name,
            create_tb_writer=args.create_tb_writer,
            files_to_copy=[args.model_config, __file__],
            random_seed=42,
            cudnn_benchmark=args.cudnn_benchmark,
            tensorboard_dir=args.tensorboard_dir + "/" + name,
        )
        args.num_gpus = neural_factory.world_size

        args.checkpoint_dir = neural_factory.checkpoint_dir

        if args.local_rank is not None:
            logging.info("Doing ALL GPU")

        # build dags
        (train_loss, callbacks, steps_per_epoch, loss_test, logits_test, label_test,) = create_all_dags(
            args, neural_factory
        )

        # train model
        neural_factory.train(
            tensors_to_optimize=[train_loss],
            callbacks=callbacks,
            lr_policy=CosineAnnealing(
                args.num_epochs * steps_per_epoch, warmup_steps=0.1 * args.num_epochs * steps_per_epoch,
            ),
            optimizer=args.optimizer,
            optimization_params={
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "betas": (args.beta1, args.beta2),
                "weight_decay": args.weight_decay,
                "grad_norm_clip": None,
            },
            batches_per_step=args.iter_per_step,
            synced_batchnorm=args.synced_bn,
            synced_batchnorm_groupsize=args.synced_bn_groupsize,
        )


    if __name__ == "__main__":
        main()


We have experimented on different pooling methods, like gram based pooling, x-vector pooling and super_vector which 
is combination of gram and x-vector. To experiment on these change pool_mode in config file accordingly.

.. note::
    This script on average for 417 hrs of data should finish 25 epochs in about 7-8 hours on Quadro GV100.

.. tip::
    To improve your embeddings performance:
        (1) Add more data and Train longer (100 epochs)
        (2) Try adding the augmentation --see config file
        (3) Use larger model
        (4) Train on several GPUs and use mixed precision (on NVIDIA Volta and Turing GPUs)
        (5) Start with pre-trained checkpoints

The above command will save the checkpoints, tensorboard logs and nemo logging files with <exp_name> under <work_dir> directory
as 

.. code-block:: bash

    <work_dir>/
    <work_dir/checkpoints/<exp_name>
    <work_dir/tensorboard/<exp_name>
    <work_dir/<log_dir>

    

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

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/speaker_recognition/speaker_reco.py ...

.. note::
    Because mixed precision requires Tensor Cores it only works on NVIDIA Volta and Turing based GPUs

Large Training Example
~~~~~~~~~~~~~~~~~~~~~~

Please refer to the `<nemo_git_repo_root>/examples/speaker_recognition/speaker_reco.py` for comprehensive example.
It builds one train DAG, one validation DAG and a test DAG to evaluate on different datasets.

Assuming, you are working with Volta-based DGX, you can run train like this:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/speaker_recognition/speaker_reco.py
    --num_epochs=25 --model_config="</configs/quartznet_spkr_5x1x512_xvector.yaml" --emb_size=1024 \
    --eval_datasets './myExps/aishell/dev_manifest.json' './myExps/voxceleb/dev_manifest.json' \
    --train_dataset='./myExps/aishell/train_manifest.json,./myExps/voxceleb/train_manifest.json' \
    --checkpoint_dir='./myExps/checkpoints/' --print_freq=400 --synced_bn \
    --checkpoint_save_freq=1000 --create_tb_writer  --eval_freq=1000  \
    --exp_name='quartznet5x1x512'  --iter_per_step=1  \
    --lr=0.02  --lr_policy='CosineAnnealing' --eval_batch_size=64 \
    --tensorboard_dir='./myExps/tensorboard/'  --warmup_steps=1000  \
    --weight_decay=0.001 --work_dir='./myExps/' --amp_opt_level=O1

The command above should trigger <num_gpus>-GPU training with mixed precision. In the command above various manifests (.json) files are various datasets. Substitute them with the ones containing your data.

.. tip::
    You can pass several manifests (comma-separated) to train on a combined dataset like this: `--train_manifest=/manifests/<first dataset>.json,/manifests/<second dataset>.json`


Fine-tuning
-----------
Training time can be dramatically reduced if starting from a good pre-trained model:

    (1) Obtain pre-trained model (jasper_encoder, jasper_decoder and configuration files).
    (2) load pre-trained weights right after you've instantiated your jasper_encoder and jasper_decoder, like this:

.. code-block:: python

    jasper_encoder.restore_from("<path_to_checkpoints>/JasperEncoder-STEP-87300.pt")
    jasper_decoder.restore_from("<path_to_checkpoints>/JasperDecoderForSpkrClass-STEP-87300.pt")
    # in case of distributed training add args.local_rank
    jasper_decoder.restore_from("<path_to_checkpoints>/JasperDecoderForSpkrClass-STEP-87300.pt", args.local_rank)

.. tip::
    When fine-tuning, use smaller learning rate.


Getting Speaker Embeddings
------------------------------  

Now that we trained a good speaker recognition model. From here we can take just pretrained encoder and finetune as mentioned above for 
various speakers (dev set) and do speaker recognition and or extract pretrained embeddings for new datasets for speaker verification tasks. Below python code shows
how we can use neural_factory infer to get embeddings from pretrained network. 

.. note::

    Before proceeding, make sure you have followed above mentioned data_preparation steps for new datasets and saved 
    checkpoints in <checkpoint> folder with given <exp_name> 

once done running below python code on a single GPU extracts embeddings to your <work_dir/embeddings> directory based on your 
evaluation dataset name as `npy` files. This will generate embeddings with test_all.npy and corresponsing filenames in 
test_all_labels.npy. 

.. code-block:: bash 
    
    python spkr_get_emb.py --model_config="./configs/quartznet_spkr_3x2x512_xvector.yaml" --num_epochs=50 \
    --emb_size=1024 --eval_datasets='<data_root>/test/test_all.json' \
    --checkpoint_dir='./myExps/checkpoints/'  \
    --exp_name='quartznet3x2x512_himia'  --iter_per_step=1 --eval_batch_size=128 \
    --work_dir='./myExps/'

.. code-block:: python

    # Copyright 2020 NVIDIA. All Rights Reserved.
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
    import copy
    import json
    import os

    import numpy as np
    from ruamel.yaml import YAML

    import nemo
    import nemo.collections.asr as nemo_asr
    import nemo.utils.argparse as nm_argparse

    logging = nemo.logging


    def parse_args():
        parser = argparse.ArgumentParser(
            parents=[nm_argparse.NemoArgParser()], description='SpeakerRecognition', conflict_handler='resolve',
        )
        parser.set_defaults(
            checkpoint_dir=None,
            optimizer="novograd",
            batch_size=32,
            eval_batch_size=64,
            lr=0.01,
            weight_decay=0.001,
            amp_opt_level="O0",
            create_tb_writer=True,
        )

        # Overwrite default args
        parser.add_argument(
            "--num_epochs",
            type=int,
            default=None,
            required=True,
            help="number of epochs to train. You should specify either num_epochs or max_steps",
        )
        parser.add_argument(
            "--model_config", type=str, required=True, help="model configuration file: model.yaml",
        )

        # Create new args
        parser.add_argument("--exp_name", default="SpkrReco_GramMatrix", type=str)
        parser.add_argument("--beta1", default=0.95, type=float)
        parser.add_argument("--beta2", default=0.5, type=float)
        parser.add_argument("--warmup_steps", default=1000, type=int)
        parser.add_argument("--load_dir", default=None, type=str)
        parser.add_argument("--synced_bn", action='store_true', help="Use synchronized batch norm")
        parser.add_argument("--synced_bn_groupsize", default=0, type=int)
        parser.add_argument("--emb_size", default=256, type=int)
        parser.add_argument("--print_freq", default=256, type=int)

        args = parser.parse_args()
        if args.max_steps is not None:
            raise ValueError("QuartzNet uses num_epochs instead of max_steps")

        return args


    def construct_name(name, lr, batch_size, num_epochs, wd, optimizer, emb_size):
        return "{0}-lr_{1}-bs_{2}-e_{3}-wd_{4}-opt_{5}-embsize_{6}".format(
            name, lr, batch_size, num_epochs, wd, optimizer, emb_size
        )


    def create_all_dags(args, neural_factory):
        '''
        creates train and eval dags as well as their callbacks
        returns train loss tensor and callbacks'''

        # parse the config files
        yaml = YAML(typ="safe")
        with open(args.model_config) as f:
            spkr_params = yaml.load(f)

        sample_rate = spkr_params['sample_rate']

        # Calculate num_workers for dataloader
        total_cpus = os.cpu_count()
        cpu_per_traindl = max(int(total_cpus / neural_factory.world_size), 1)

        # create separate data layers for eval
        # we need separate eval dags for separate eval datasets
        # but all other modules in these dags will be shared

        eval_dl_params = copy.deepcopy(spkr_params["AudioToSpeechLabelDataLayer"])
        eval_dl_params.update(spkr_params["AudioToSpeechLabelDataLayer"]["eval"])
        del eval_dl_params["train"]
        del eval_dl_params["eval"]
        eval_dl_params['shuffle'] = False  # To grab  the file names without changing data_layer

        data_layer_test = nemo_asr.AudioToSpeechLabelDataLayer(
            manifest_filepath=args.eval_datasets[0],
            labels=None,
            batch_size=args.batch_size,
            num_workers=cpu_per_traindl,
            **eval_dl_params,
            # normalize_transcripts=False
        )
        # create shared modules

        data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate, **spkr_params["AudioToMelSpectrogramPreprocessor"],
        )

        # (QuartzNet uses the Jasper baseline encoder and decoder)
        encoder = nemo_asr.JasperEncoder(**spkr_params["JasperEncoder"],)

        decoder = nemo_asr.JasperDecoderForSpkrClass(
            feat_in=spkr_params['JasperEncoder']['jasper'][-1]['filters'],
            num_classes=254,
            emb_sizes=spkr_params['JasperDecoderForSpkrClass']['emb_sizes'].split(','),
            pool_mode=spkr_params["JasperDecoderForSpkrClass"]['pool_mode'],
        )

        # --- Assemble Validation DAG --- #
        audio_signal_test, audio_len_test, label_test, _ = data_layer_test()

        processed_signal_test, processed_len_test = data_preprocessor(
            input_signal=audio_signal_test, length=audio_len_test
        )

        encoded_test, _ = encoder(audio_signal=processed_signal_test, length=processed_len_test)

        _, embeddings = decoder(encoder_output=encoded_test)

        return embeddings, label_test


    def main():
        args = parse_args()

        print(args)

        name = construct_name(
            args.exp_name, args.lr, args.batch_size, args.num_epochs, args.weight_decay, args.optimizer, args.emb_size
        )
        work_dir = name
        if args.work_dir:
            work_dir = os.path.join(args.work_dir, name)

        # instantiate Neural Factory with supported backend
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch,
            local_rank=args.local_rank,
            optimization_level=args.amp_opt_level,
            log_dir=work_dir,
            checkpoint_dir=args.checkpoint_dir + "/" + args.exp_name,
            create_tb_writer=False,
            files_to_copy=[args.model_config, __file__],
            random_seed=42,
            cudnn_benchmark=args.cudnn_benchmark,
        )
        args.num_gpus = neural_factory.world_size

        args.checkpoint_dir = neural_factory.checkpoint_dir

        if args.local_rank is not None:
            logging.info('Doing ALL GPU')

        # build dags
        embeddings, label_test = create_all_dags(args, neural_factory)

        eval_tensors = neural_factory.infer(tensors=[embeddings, label_test], checkpoint_dir=args.checkpoint_dir)
        # inf_loss , inf_emb, inf_logits, inf_label = eval_tensors
        inf_emb, inf_label = eval_tensors
        whole_embs = []
        whole_labels = []
        manifest = open(args.eval_datasets[0], 'r').readlines()

        for line in manifest:
            line = line.strip()
            dic = json.loads(line)
            filename = dic['audio_filepath'].split('/')[-1]
            whole_labels.append(filename)

        for idx in range(len(inf_label)):
            whole_embs.extend(inf_emb[idx].numpy())

        embedding_dir = args.work_dir + './embeddings/'
        if not os.path.exists(embedding_dir):
            os.mkdir(embedding_dir)

        filename = os.path.basename(args.eval_datasets[0]).split('.')[0]
        name = embedding_dir + filename

        np.save(name + '.npy', np.asarray(whole_embs))
        np.save(name + '_labels.npy', np.asarray(whole_labels))
        logging.info("Saved embedding files to {}".format(embedding_dir))


    if __name__ == '__main__':
        main()

.. note::
    If you are working on a different dataset, make sure to change num_classes argument in JasperDecoderForSpkrClass 
    based on number of pretrained speakers.

SCORING
-------

Though speaker verification scoring is slightly dependent on how we get the trial-files. So this evaluattion script may
not work well without slight modifications on your challange/dataset trial file. Here we provide a script scoring
on hi-mia :cite:`speaker-tut-himia` whose trial file has structure <speaker_name1> <speaker_name2> <target/nontarget> 

Once your embeddings are prepared in <embeddings_dir> , the below command would output the EER% based on cosine similarity score. 
script to this is found in <nemo>/scripts. Make sure trails file is placed in <embeddings_dir>

.. code-block:: bash

    python hi-mia_eval.py --data_root='<embeddings_dir' --emb='<emb_dir>/test_all.npy' --emb_labels='<emb_dir>/test_all_labels.npy' --emb_size 1024

This should output an EER rate of 8.72%. Above script also generates all_embs_himia.npy file which can be later used during PLDA scoring.
.. Here the --task argument was ffsvc task id for challenge. 

We also used PLDA backend to finetune our speaker embeddings furthur. We used kaldi PLDA scripts to train PLDA and evaluate as well. 
so from this point going forward, please make sure you installed kaldi and was added to your path as KALDI_ROOT. 

.. note::
    If you would like to train PLDA on a <set>, please make sure you generated embeddings for those all well by following above 
    mentioned procedure. And also corresponding spk2utt and utt2spk files in '<work_dir>/embeddings/' directory. We already
    generated utt2spk file and can be found in <data_root>/{set} . Then running kaldi binary 
    utt2spk_to_spk2utt.pl generates spk2utt file as well. Also please copy trails_1m file from <data_root> to '<work_dir>/embeddings/' for PLDA training.

We provide two scripts that makes data preparation for kaldi processing and evaluation. To process data in kaldi format run below script with arguments as shown below :

.. code-block:: python
       
        python kaldi_plda.py --root=''<embedding_dir>'  --train_embs='<embedding_dir>/train.npy' --train_labels='<embedding_dir>/train_labels.npy'  
        --eval_embs='<embedding_dir>/all_embs_himia.npy' --eval_labels='<embedding_dir>/all_ids_himia.npy' --stage=1

Here --stage = 1 trains PLDA model but if you already have a trained PLDA then you can directly evaluate on it by --stage=2 option. 

This should output an EER of 6.32% with minDCF: 0.455

References
----------

.. bibliography:: speaker.bib
    :style: plain
    :labelprefix: SPEAKER-TUT
    :keyprefix: speaker-tut-
