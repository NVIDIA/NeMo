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
from nemo.utils import logging


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
