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

        audio_signal_test, audio_len_test, label_test, label_len_test = eval_layer()
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
