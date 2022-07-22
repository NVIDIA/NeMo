# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import pathlib

import omegaconf
import torch
import wget

try:
	import fairseq
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("This script requires fairseq to be installed to load Wav2Vec modules. Please see https://github.com/facebookresearch/fairseq for installation instructions.")

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.utils import logging

"""
Script to load Fairseq Wav2Vec2.0 style pretrained model weights into NeMo models. Requires Fairseq installation.
Arguments:
	--fairseq_model:
		Name of Fairseq model to download. Currently supported models are:
			Wav2Vec2.0 Base Model trained on 960h LibriSpeech corpus: 'base'
			Wav2Vec2.0 Large Model trained on 960h LibriSpeech corpus: 'large'
			Wav2Vec2.0 Large Model trained on LibriVox 60kh corpus: 'lv-60'
			Wav2Vec2.0 Large Model trained on xlsr-53 corpora: 'xlsr'
	--fairseq_path:
		Directory location of Fairseq model. If no model is present in the directory, one will be downloaded from
		the relevant url. If no directory is provided, searches current working directory.
	--nemo_path:
		Directory to save nemo_model to. If none is provided, saves to current working directory.
	--modules:
		List of modules to transfer Fairseq weights from. Avilable: ['feature_extractor', 'encoder', 'quantizer', 'all']. 
		Default 'all'

NOTE: This script only instantiates the full weights from the feature extractor, encoder, and quantizer portions of
	the Fairseq models. Decoder weights cannot be instantiated due to architecture limitations. In most cases, this
	will have no effect on finetuning and other applications.

For further information on these models, please see their source page: https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec
"""

# Urls for acquiring models
model_url = {
    'base': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt',
    'large': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt',
    'lv-60': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt',
    'lv-60-r': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv.pt',
    'xlsr': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt',
}

# Dictionaries for renaming modules
preprocessor_key_terms = {
    'feature_extractor.': '',
    '.2.1.bias': '.1.1.bias',
    '.2.1.weight': '.1.1.weight',
    '.0.2.bias': '.0.1.bias',
    '.0.2.weight': '.0.1.weight',
}
encoder_key_terms = {
    'encoder.': '',
    'self_attn.k_proj': 'first_sub_layer.key_net',
    'self_attn.v_proj': 'first_sub_layer.value_net',
    'self_attn.q_proj': 'first_sub_layer.query_net',
    'self_attn.out_proj': 'first_sub_layer.out_projection',
    'self_attn_layer_norm': 'layer_norm_2',
    'fc1': 'second_sub_layer.dense_in',
    'fc2': 'second_sub_layer.dense_out',
    'final_layer_norm': 'layer_norm_1',
}


parser = argparse.ArgumentParser(
    description="Downloads Fairseq version of Wav2Vec2.0 style pretrained model and transfers weights to NeMo equivalent."
)

parser.add_argument(
    '--fairseq_model',
    type=str,
    default='base',
    choices=model_url.keys(),
    help='name of model to download from Fairseq',
)
parser.add_argument('--fairseq_path', type=str, default=None, help='desired directory to save/retrieve fariseq model')
parser.add_argument('--nemo_path', type=str, default=".", help='desired directory to save NeMo model')
parser.add_argument(
    '--modules',
    type=str,
    nargs='+',
    default=['all'],
    choices=['feature_extractor', 'encoder', 'quantizer', 'all'],
    help='desired modules to transfer fairseq weights from into their NeMo equivalents',
)

args = parser.parse_args()


def get_nemo_model(model_name: str):
    # Instantiates NeMo model
    logging.info("Creating NeMo model")
    if model_name == "base":
        config_name = 'wav2vec_pretrain.yaml'
    else:  # All other models use the large config
        config_name = 'wav2vec_pretrain_large.yaml'
    cfg = omegaconf.OmegaConf.load(f"../conf/ssl/wav2vec/{config_name}")

    # Changing cfg information to prevent errors
    logging.info("Modifying NeMo configs")
    cfg.model.train_ds.manifest_filepath = None
    cfg.model.validation_ds.manifest_filepath = None

    cfg.model.train_ds.batch_size = 1
    cfg.model.validation_ds.batch_size = 1

    if model_name == 'large':
        # Fairseq's original large implementation is slightly off than more current models
        cfg.model.preprocessor.conv_bias = False
        cfg.model.preprocessor.extractor_mode = "group_norm"
    logging.info(f"Instantiating NeMo version of Wav2Vec '{model_name}' model")
    return SpeechEncDecSelfSupervisedModel(cfg=cfg.model)


def get_fair_model(model_name, model_path=None):
    # Loads weights from fairseq model
    logging.info(f"Loading Fairseq Wav2Vec '{model_name}' model")
    url = model_url[model_name]

    wav2vec_file = os.path.basename(url)
    wav2vec_path = os.path.join(model_path, wav2vec_file) if model_path else wav2vec_file

    if not pathlib.Path(wav2vec_path).is_file():  # Checks for file, if not here, downloads to location
        logging.info(f"Fairseq model not found. Downloading '{model_name}' to {wav2vec_path}")
        wget.download(url, out=wav2vec_path)

    return torch.load(wav2vec_path)


def main():
    # Loads models
    nemo_model = get_nemo_model(args.fairseq_model)
    fairseq_model = get_fair_model(args.fairseq_model, args.fairseq_path)

    # To ensure all weights are being loaded, state_dicts are copied iteratively by module
    modules = set(args.modules)
    preprocessor = 'all' in modules or 'feature_extractor' in modules
    encoder = 'all' in modules or 'encoder' in modules
    quantizer = 'all' in modules or 'quantizer' in modules

    # Preprocessors
    leftovers = fairseq_model['model']
    if preprocessor:
        logging.info(f"Copying {args.fairseq_model} weights to NeMo preprocessor module")
        preprocessor_dict, leftovers = load_weights(
            leftovers, nemo_model.preprocessor.state_dict(), key_terms=preprocessor_key_terms
        )
        nemo_model.preprocessor.load_state_dict(preprocessor_dict, strict=True)

    if encoder:
        logging.info(f"Copying {args.fairseq_model} weights to NeMo encoder module")
        encoder_dict, leftovers = load_weights(leftovers, nemo_model.encoder.state_dict(), key_terms=encoder_key_terms)
        nemo_model.encoder.load_state_dict(encoder_dict, strict=True)

    if quantizer:
        logging.info(f"Copying {args.fairseq_model} weights to NeMo loss module")
        loss_dict, leftovers = load_weights(leftovers, nemo_model.loss.state_dict())
        nemo_model.loss.load_state_dict(loss_dict, strict=True)

    path = os.path.join(args.nemo_path, f"wav2vec_{args.fairseq_model}.nemo")
    logging.info(f"Saving NeMo model to {path}")
    nemo_model.save_to(path)


def load_weights(source, sink, key_terms={}, key_words={}):
    # Loads weight from given source dictionary that align with constraints in sink dictionary

    dict_weights = {}
    leftovers = {}  # For keeping remaining weights and reducing repetition

    for key in source.keys():
        new_key = None
        if key in sink:  # Exists in sink dictionary, we store
            dict_weights[key] = source[key]
        elif key in key_words:  # Known pairing is found, store weight as new naem
            new_key = key_words[key]
            dict_weights[new_key] = source[key]
        else:  # Iterates through key terms and change key names to align with desired keys
            for term in key_terms:
                if term in key:
                    if new_key:  # Checks if we've began altering the key yet
                        new_key = new_key.replace(term, key_terms[term])
                    else:
                        new_key = key.replace(term, key_terms[term])
            if new_key:
                dict_weights[new_key] = source[key]
            else:
                leftovers[key] = source[key]  # Store remaining weights
    return dict_weights, leftovers


if __name__ == "__main__":
    main()
